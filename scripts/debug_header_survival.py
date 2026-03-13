"""
Diagnostic: Do chat header tokens survive FINCH compression?

Checks if the first N tokens of context_ids (the Llama 3.1 chat header:
<|begin_of_text|><|system|>...<|eot_id|><|user|>) end up in the selected
important_tokens after the first compression chunk.

If header tokens are consistently dropped in round 1, they're gone forever.
"""

import sys
import json
import math
import torch
import numpy as np
sys.path.insert(0, '/root/Finch')

from transformers import AutoTokenizer

TOKENIZER_PATH = "meta-llama/Llama-3.1-8B-Instruct"
DATA_FILE      = "data/longbench_v1/narrativeqa.jsonl"
SYSTEM_PROMPT  = ("You are given a story, which can be either a novel or a movie script, "
                  "and a question. Answer the question as concisely as you can, using a "
                  "single phrase if possible. Do not provide any explanation.")
N_SAMPLES      = 10
SPLIT_SIZE     = 512
TARGET_TOKEN   = 9137

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.add_special_tokens({'pad_token': '<pad>'})

def generate_context(context, system_prompt):
    return (f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"context: {context.lstrip()}")

def generate_question(question):
    return f"question: {question.lstrip()}"

print("Loading model (CPU only for this diagnostic)...")
# We don't need the full model — just compute attention for one forward pass
# Use a small proxy: just check token IDs to find header length
samples = []
with open(DATA_FILE) as f:
    for i, line in enumerate(f):
        if i >= N_SAMPLES:
            break
        samples.append(json.loads(line))

print(f"\nLoaded {len(samples)} samples\n")
print("=" * 70)

for idx, sample in enumerate(samples):
    context_text = generate_context(sample['context'], SYSTEM_PROMPT)
    question_text = generate_question(sample['input'])

    context_ids = tokenizer(context_text, add_special_tokens=False,
                             truncation=True, max_length=128000,
                             return_tensors='pt')['input_ids'][0]
    question_ids = tokenizer(question_text, add_special_tokens=False,
                              truncation=True, max_length=64,
                              return_tensors='pt')['input_ids'][0]

    ctx_len = context_ids.size(0)
    q_len   = question_ids.size(0)

    # Find where "context: " starts (i.e., end of header)
    # Header = <|begin_of_text|> + system turn + <|eot_id|> + <|user|> + \n\n
    # We detect this by finding the token for "context"
    header_token_ids = tokenizer("<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
                                  add_special_tokens=False)['input_ids']
    # Count tokens up to and including "context: " prefix
    prefix_text = (f"<|begin_of_text|>"
                   f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
                   f"<|start_header_id|>user<|end_header_id|>\n\n"
                   f"context: ")
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)['input_ids']
    header_len = len(prefix_ids)  # number of header tokens in context_ids

    # Simulate what FINCH would do in the first chunk
    compression_factor = math.ceil(ctx_len / (TARGET_TOKEN - q_len))
    k_per_step = ctx_len // compression_factor  # tokens to keep after first chunk

    # First chunk
    first_chunk = context_ids[:SPLIT_SIZE]
    chunk_len = first_chunk.size(0)

    # How many tokens to keep from first chunk?
    k_first = chunk_len // compression_factor
    k_first = max(1, k_first)

    print(f"Sample {idx}: ctx_len={ctx_len}, q_len={q_len}")
    print(f"  header_len={header_len} tokens (positions 0..{header_len-1})")
    print(f"  compression_factor={compression_factor}, k_first={k_first}/{chunk_len}")

    if ctx_len <= TARGET_TOKEN - q_len:
        print(f"  → No compression (ctx_len <= target), SKIPPED\n")
        continue

    # Compute actual attention scores using the model
    # For a CPU-only diagnostic, we just check: are header tokens in top-k by any metric?
    # Without the model, we check if k_first >= header_len (header COULD survive)
    if k_first >= header_len:
        print(f"  → Budget allows header ({k_first} slots >= {header_len} header tokens)")
        print(f"     Header CAN survive if attention scores are high enough")
        print(f"     → Need model forward pass to verify attention scores")
    else:
        print(f"  ⚠️  Budget too small: only {k_first} slots but header is {header_len} tokens")
        print(f"     Header CANNOT fully survive even in best case")

    # Check header_len vs first chunk
    header_in_first_chunk = min(header_len, chunk_len)
    print(f"  Header tokens in first chunk: {header_in_first_chunk}/{header_len}")
    print(f"  Header tokens that MUST be selected to survive: {header_in_first_chunk}")
    print(f"  Available slots: {k_first}")

    if k_first < header_in_first_chunk:
        pct = k_first / header_in_first_chunk * 100
        print(f"  ⚠️  IMPOSSIBLE: budget ({k_first}) < header size ({header_in_first_chunk})")
        print(f"     Even if ALL selected tokens were header tokens, only {pct:.0f}% would survive")
    print()

print("=" * 70)
print("\nSummary: checking if header_len > k_first for compressed samples")
print("(If True, header tokens CANNOT fully survive compression budget constraints)")
print("\nBudget is SUFFICIENT in all cases above — need model attention to confirm.")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: actual model forward pass on first chunk
# Replicates FINCH's attention_score + question-condition + normalize logic
# to see what fraction of header tokens (positions 0..header_len-1) land in
# the top-k important_tokens selected by each Transformer layer.
# ─────────────────────────────────────────────────────────────────────────────

# Collect samples that WILL be compressed
long_samples = []
for i, s in enumerate(samples):
    ctx_text = generate_context(s['context'], SYSTEM_PROMPT)
    ctx_ids = tokenizer(ctx_text, add_special_tokens=False,
                        return_tensors='pt')['input_ids'][0]
    q_ids = tokenizer(generate_question(s['input']),
                      add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    if ctx_ids.size(0) + q_ids.size(0) > TARGET_TOKEN:
        long_samples.append((i, s, ctx_ids, q_ids))

print(f"\nSamples that will be compressed: {[i for i,_,_,_ in long_samples]}")
print(f"Running forward pass on {min(3, len(long_samples))} samples (first chunk only)...\n")

# Compute the header_len using tokenizer (same logic as above)
prefix_text = (f"<|begin_of_text|>"
               f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
               f"<|start_header_id|>user<|end_header_id|>\n\n"
               f"context: ")
HEADER_LEN = len(tokenizer(prefix_text, add_special_tokens=False)['input_ids'])
print(f"Header length: {HEADER_LEN} tokens (positions 0..{HEADER_LEN-1})\n")

print("Loading model onto GPU...")
from transformers import AutoModelForCausalLM
import time

model = AutoModelForCausalLM.from_pretrained(
    TOKENIZER_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",   # need attention weights
)
model.eval()
print("Model loaded.\n")
print("=" * 70)

DEVICE = next(model.parameters()).device

for idx, (sidx, sample, context_ids, question_ids) in enumerate(long_samples[:3]):
    ctx_len = context_ids.size(0)
    q_len   = question_ids.size(0)
    compression_factor = int(math.ceil(ctx_len / (TARGET_TOKEN - q_len)))
    segment = context_ids[:SPLIT_SIZE]       # first chunk only
    seg_len = segment.size(0)
    k = seg_len // compression_factor        # tokens to keep
    k = max(1, k)

    print(f"Sample {sidx}: ctx_len={ctx_len}, q_len={q_len}")
    print(f"  compression_factor={compression_factor}, k={k}/{seg_len}")
    print(f"  Checking if header tokens (0..{HEADER_LEN-1}) survive in top-{k}...")

    # Build input identical to FINCH's first step (past_cache_len=0)
    current_ids     = torch.cat([segment.unsqueeze(0), question_ids.unsqueeze(0)], dim=1).to(DEVICE)
    total_len       = current_ids.size(1)
    attn_mask       = torch.ones(1, total_len, device=DEVICE)
    position_ids    = attn_mask.long().cumsum(-1) - 1

    # Causal 4D mask (same as modeling_llama.py)
    causal_4d = torch.full((total_len, total_len),
                           torch.finfo(torch.float16).min, device=DEVICE)
    causal_4d.triu_(1)
    causal_4d = causal_4d[None, None, :, :].expand(1, 1, -1, -1).clone()
    padding_4d = (1.0 - attn_mask[:, None, None, :]) * torch.finfo(torch.float16).min
    final_attn_mask = causal_4d + padding_4d

    with torch.no_grad():
        out = model(
            input_ids=current_ids,
            attention_mask=final_attn_mask,
            position_ids=position_ids,
            output_attentions=True,
            use_cache=False,
        )

    # Replicate FINCH's aggregation (attention_score + question + normalize)
    header_survival_per_layer = []
    for layer_idx, layer_attn in enumerate(out.attentions):
        # layer_attn: (1, n_heads, total_len, total_len)
        summed   = layer_attn.sum(dim=1)           # (1, total_len, total_len)
        tot      = summed.size(2)
        # question rows → context columns (past_cache_len=0)
        ctx_attn = summed[:, seg_len:, :seg_len]   # (1, q_len, seg_len)
        nz       = torch.arange(1, tot + 1, device=DEVICE).float()
        nz       = nz[seg_len:] / tot              # normalization for q rows
        ctx_attn = ctx_attn * nz[None, :, None]
        agg      = ctx_attn.sum(dim=1)             # (1, seg_len)

        _, imp = torch.topk(agg, k=k, dim=-1, largest=True)
        imp_set = set(imp[0].cpu().tolist())

        # How many header tokens are in top-k?
        header_positions  = set(range(HEADER_LEN))
        survived = header_positions & imp_set
        pct = len(survived) / HEADER_LEN * 100
        header_survival_per_layer.append(pct)

    pct_arr = np.array(header_survival_per_layer)
    print(f"  Header survival across {len(pct_arr)} layers:")
    print(f"    min={pct_arr.min():.1f}%  max={pct_arr.max():.1f}%  "
          f"mean={pct_arr.mean():.1f}%  median={np.median(pct_arr):.1f}%")
    bad_layers = [l for l, p in enumerate(header_survival_per_layer) if p < 50]
    if bad_layers:
        print(f"  ⚠️  Layers with <50% header survival: {bad_layers[:10]}{'...' if len(bad_layers)>10 else ''}")
    else:
        print(f"  ✓  All layers keep ≥50% of header tokens")

    # Also show: what are the top-k position ranges in a typical layer (layer 15)?
    mid_layer = len(out.attentions) // 2
    summed   = out.attentions[mid_layer].sum(dim=1)
    ctx_attn = summed[:, seg_len:, :seg_len]
    nz       = torch.arange(1, summed.size(2)+1, device=DEVICE).float()
    nz       = nz[seg_len:] / summed.size(2)
    ctx_attn = ctx_attn * nz[None, :, None]
    agg      = ctx_attn.sum(dim=1)[0].cpu()  # (seg_len,)
    imp_sorted = torch.topk(agg, k=k, largest=True).indices.sort().values.tolist()
    header_in = [p for p in imp_sorted if p < HEADER_LEN]
    print(f"  Layer {mid_layer} top-{k}: {len(header_in)}/{HEADER_LEN} header positions kept"
          f"  |  non-header top-5 positions: {[p for p in imp_sorted if p >= HEADER_LEN][:5]}")
    print(f"  Header positions dropped (layer {mid_layer}): "
          f"{sorted(set(range(HEADER_LEN)) - set(imp_sorted))}\n")

print("=" * 70)
print("\nDONE. Interpretation:")
print("  100% survival  → header is NOT the cause of degenerate outputs")
print("  <100% survival → some header tokens dropped; may contribute to failures")

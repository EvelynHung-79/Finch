# context-compression
> **Note**  
> The **official implementation of FINCH (Prompt-guided Key-Value Cache Compression)** is available in the **KVPress** library:
>
> - **FINCH (FinchPress) implementation:**  
>   https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/finch_press.py
>
> - **Chunked version & discussion (PR #64):**  
>   https://github.com/NVIDIA/kvpress/pull/64
>
> The KVPress implementation matches the authors’ reference code and has been validated to produce **bit-exact results**.

## Paper

Official implementation of the TACL paper:

**FINCH: Prompt-guided Key-Value Cache Compression**  
https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280/FINCH-Prompt-guided-Key-Value-Cache-Compression

If you use FINCH / FinchPress, please cite the paper.

```bibtex
@article{10.1162/tacl_a_00716,
    author = {Corallo, Giulio and Papotti, Paolo},
    title = {FINCH: Prompt-guided Key-Value Cache Compression for Large Language Models},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {12},
    pages = {1517-1532},
    year = {2024},
    month = {11},
    abstract = {Recent large language model applications, such as Retrieval-Augmented Generation and chatbots, have led to an increased need to process longer input contexts. However, this requirement is hampered by inherent limitations. Architecturally, models are constrained by a context window defined during training. Additionally, processing extensive texts requires substantial GPU memory. We propose a novel approach, Finch, to compress the input context by leveraging the pre-trained model weights of the self-attention. Given a prompt and a long text, Finch iteratively identifies the most relevant Key (K) and Value (V) pairs over chunks of the text conditioned on the prompt. Only such pairs are stored in the KV cache, which, within the space constrained by the context window, ultimately contains a compressed version of the long text. Our proposal enables models to consume large inputs even with high compression (up to 93x) while preserving semantic integrity without the need for fine-tuning.},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00716},
    url = {https://doi.org/10.1162/tacl_a_00716},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00716/2480391/tacl_a_00716.pdf},
}
```

## Quickstart (use KVPress)

Install KVPress and use `FinchPress`:

```bash
pip install kvpress
```
## Minimal Example
> **Why this is different**: FINCH requires inserting a special **delimiter token** between the `context` and the `question`.
> You must: (1) create `FinchPress`, (2) call `press.update_model_and_tokenizer(...)`, (3) append `press.delimiter_token` and the `question` to the `context`, and (4) pass an empty `question` to the pipeline.
```python
from transformers import pipeline
from kvpress import FinchPress
import torch

# 1) Build the KVPress generation pipeline
device = "cuda:0"  # or "auto" / "cpu"
model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_kwargs = {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}

pipe = pipeline(
    "kv-press-text-generation",
    model=model,
    device=device,
    model_kwargs=model_kwargs,
    trust_remote_code=True,
)

# 2) Prepare your data
context = "A very long text you want to compress once and for all"
question = "\nA question about the compressed context"

# 3) Configure FINCH
press = FinchPress(
    compression_ratio=0.5,   # keep 50% of tokens (set per your budget)
    normalize_scores=True,   # recommended
)

# 4) FINCH requires adding a delimiter token between context and question,
#    and updating the model/tokenizer so the delimiter is recognized.
press.update_model_and_tokenizer(pipe.model, pipe.tokenizer)

# 5) Append the delimiter + question to the context and pass an empty `question` to the pipeline
augmented_context = context + press.delimiter_token + question

result = pipe(
    augmented_context,
    question="",          # FINCH expects the question to be inside the context
    press=press,
    max_new_tokens=128,   # tune per task
)

answer = result["answer"]
print(answer)
```
> For large contexts, you may also enable the chunked variant of FinchPress. See this PR for details: https://github.com/NVIDIA/kvpress/pull/64

For full usage, benchmarks, and configuration options, refer to the KVPress repository.
## Development (this repo)
This repository is kept for archival/reference. For active development and up-to-date implementations, use `KVPress`.

## (Legacy) Development Installation
```bash
git clone git@github.com:anonymous/context-compression.git
cd context-compression
pre-commit install
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t image-context-compression -f docker/Dockerfile .
docker run --gpus all --detach -v /path/to/context-compression:/home/jovyan/context-compression image-context-compression tail -f /dev/null
docker exec -it <container_id> /bin/bash
```
Re-install in edit mode:
```
pip install -e .[dev]
```

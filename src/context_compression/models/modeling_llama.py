import math
import time
from typing import Optional, Tuple, Dict, Any, List
import torch
from transformers.cache_utils import DynamicCache

class ModifiedDynamicCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        self.key_cache = []
        self.value_cache = []
        self.cos_sin_cache = []
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        idx = layer_idx if layer_idx is not None else 0
        if len(self.key_cache) <= idx:
            # print(f"[DEBUG] get_seq_length(layer={idx}) -> cache empty, returning 0")
            return 0
        length = self.key_cache[idx].shape[-2]
        # print(f"[DEBUG] get_seq_length(layer={idx}) -> returning {length}")
        return length

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        length = self.get_seq_length(layer_idx)
        # print(f"[DEBUG] get_usable_length(new_seq={new_seq_length}, layer={layer_idx}) -> returning {length}")
        return length

    @property
    def _seen_tokens(self):
        # print(f"[DEBUG] _seen_tokens property accessed! Returning {self.seen_tokens}")
        return self.seen_tokens
    
    @_seen_tokens.setter
    def _seen_tokens(self, value):
        pass
    # ===============================================================

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if cos.dim() == 3 and key_states.dim() == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    @staticmethod
    def _rerotate_cos_sin(cos, sin, important_pos_batch):
        original_dtype = cos.dtype
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        batch_size, seq_length = important_pos_batch.shape

        if cos.dim() == 2:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        indices = important_pos_batch.unsqueeze(-1).expand(-1, -1, cos.size(-1))
        original_cos = torch.gather(cos, 1, indices)
        original_sin = torch.gather(sin, 1, indices)

        shifted_cos = cos[:, :seq_length, :]
        shifted_sin = sin[:, :seq_length, :]

        rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
        
        idx = torch.arange(seq_length, device=important_pos_batch.device)[None, :]
        less_than_mask = (important_pos_batch < idx)

        rerotation_sin = torch.where(
            less_than_mask[:, :, None],
            original_sin * shifted_cos - original_cos * shifted_sin,
            - (original_sin * shifted_cos - original_cos * shifted_sin)
        )
        
        same_pos_mask = (important_pos_batch == idx)
        rerotation_cos[same_pos_mask] = 1
        rerotation_sin[same_pos_mask] = 0

        return rerotation_cos.to(original_dtype), rerotation_sin.to(original_dtype)

    @staticmethod
    def gather_important_tokens(states, indices):
        return torch.gather(states, 2,
                            indices.unsqueeze(1).unsqueeze(-1).expand(-1, states.size(1), -1, states.size(3)))

    def update_rope(self, layer_index, key_states, important_pos):
        # if layer_index == 0:
            # print(f"\n[DEBUG] --- update_rope() Layer 0 | Compressing to {important_pos.size(1)} tokens ---")
        
        seq_length = key_states.shape[-2]
        
        cos_cache = self.cos_sin_cache[layer_index]["cos"]
        sin_cache = self.cos_sin_cache[layer_index]["sin"]
        
        if cos_cache.dim() == 2:
            cos_sliced = cos_cache[:seq_length, :]
            sin_sliced = sin_cache[:seq_length, :]
        else:
            cos_sliced = cos_cache[:, :seq_length, :]
            sin_sliced = sin_cache[:, :seq_length, :]

        new_cos, new_sin = self._rerotate_cos_sin(cos_sliced, sin_sliced, important_pos)
        
        self.key_cache[layer_index] = self._apply_key_rotary_pos_emb(
            self.gather_important_tokens(self.key_cache[layer_index], important_pos),
            new_cos,
            new_sin
        )
        self.value_cache[layer_index] = self.gather_important_tokens(self.value_cache[layer_index], important_pos)
        self.cos_sin_cache[layer_index]["cos"] = new_cos
        self.cos_sin_cache[layer_index]["sin"] = new_sin
        self.seen_tokens = important_pos.size(1)
        
        return self.key_cache[layer_index], self.value_cache[layer_index]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # if layer_idx == 0:
            # print(f"\n[DEBUG] --- update() Layer 0 | Input seq len: {key_states.shape[-2]} ---")

        if cache_kwargs is not None:
            sin = cache_kwargs.get("sin")
            cos = cache_kwargs.get("cos")
        else:
            sin = None
            cos = None
            
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            if sin is not None and cos is not None:
                self.cos_sin_cache.append({"sin": sin, "cos": cos})
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            if sin is not None and cos is not None:
                old_sin = self.cos_sin_cache[layer_idx]["sin"]
                old_cos = self.cos_sin_cache[layer_idx]["cos"]
                self.cos_sin_cache[layer_idx] = {
                    "sin": torch.cat([old_sin, sin], dim=-2),
                    "cos": torch.cat([old_cos, cos], dim=-2)
                }
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # if layer_idx == 0:
            # print(f"[DEBUG] update() finished | Cache len is now: {self.key_cache[layer_idx].shape[-2]}")

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

import transformers.cache_utils

transformers.cache_utils.DynamicCache = ModifiedDynamicCache

from transformers import LlamaForCausalLM
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaConfig
logger = logging.get_logger(__name__)

class LlamaForCompressedCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, mode, split_size, target_token, condition="question", normalize=True, is_full=False, distance_metric=None):  # changed by GC
        config._attn_implementation = "eager"
        super().__init__(config)
        self.mode = mode
        self.target_token = target_token
        self.split_size = split_size
        self.condition = condition
        self.normalize = normalize
        self.is_full = is_full
        if distance_metric == "euclidean":
            self.p = 2
        elif distance_metric == "manhattan":
            self.p = 1
        elif distance_metric == "minkowski":
            self.p = 3
        else:
            self.p = 0
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        # [DEBUG LOG START] 檢查進入模型的資料
        if input_ids is not None:
            # 檢查是否有空的 Input
            if input_ids.numel() == 0:
                print(f"\n[DEBUG Forward] !!! WARNING: Empty input_ids detected! Shape: {input_ids.shape}")
            else:
                max_id = input_ids.max().item()
                min_id = input_ids.min().item()
                vocab_size = self.config.vocab_size
                
                # 檢查是否有非法 ID (造成 Device-side assert 的主因)
                if max_id >= vocab_size or min_id < 0:
                    print(f"\n[DEBUG Forward] CRITICAL ERROR: Input ID out of bounds!")
                    print(f"  - Shape: {input_ids.shape}")
                    print(f"  - Max ID: {max_id} (Vocab: {vocab_size})")
                    print(f"  - Min ID: {min_id}")
        # [DEBUG LOG END]

        # 呼叫原本父類別的 forward 繼續執行
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
    
    def generate(
        self,
        accelerator,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        question_ids: Optional[torch.LongTensor] = None,
        question_attention_mask: Optional[torch.FloatTensor] = None,
        context_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs
    ):

        past_key_values = ModifiedDynamicCache()
        accelerator.log({
            "context_size_mean": context_ids.size(1),
            "context_size_min": context_ids.size(1),
            "context_size_max": context_ids.size(1)
        })
        print("context size is", context_ids.size(1))
        print("input id size is", input_ids.size(-1))
        
        if context_ids.size(1) +  input_ids.size(-1) > self.target_token and not self.target_token == 4096:
            print("Compressing...")
            start_processing_time = time.time()
            if self.split_size == "auto":
                segment_length = self.target_token + generate_kwargs['max_new_tokens'] -  input_ids.size(-1)
            else:
                segment_length = self.split_size
            self.compression_factor = int(math.ceil(context_ids.size(1) / (self.target_token - input_ids.size(-1))))
            context_ids_list = torch.split(context_ids, segment_length, dim=1)
            context_attention_mask_list = torch.split(context_attention_mask, segment_length, dim=1)
            past_attention_mask = torch.zeros(context_attention_mask.size(0), 0, dtype=context_attention_mask.dtype, device=context_attention_mask.device)
            for step, (segment_context_ids, segment_attention_mask) in enumerate(zip(context_ids_list, context_attention_mask_list)):
                segment_attention_mask = torch.cat([past_attention_mask, segment_attention_mask], dim=1)
                past_cache_len = past_key_values.seen_tokens
                current_ids = torch.cat([segment_context_ids, question_ids], dim=1)
                current_attention_mask = torch.cat([segment_attention_mask, question_attention_mask], dim=1)
                position_ids = (current_attention_mask.long().cumsum(-1) - 1)
                position_ids.masked_fill_(current_attention_mask == 0, 1)  # can be filled with anything >= 0
                position_ids = position_ids[:, -current_ids.shape[1]:]

                # ================= [新增區塊] 手動建構 4D Causal Mask =================
                bsz, q_len = current_ids.shape
                
                # 1. 建立當前序列的 Causal Mask (右上方為負無限大，阻擋看未來的 token)
                causal_4d = torch.full((q_len, q_len), torch.finfo(self.model.dtype).min, device=current_ids.device)
                causal_4d.triu_(1)
                
                # 2. 如果有過去的 Cache，把它們補在前面 (全部為 0，代表可以完全看見)
                if past_cache_len > 0:
                    past_mask = torch.zeros(q_len, past_cache_len, device=current_ids.device, dtype=causal_4d.dtype)
                    causal_4d = torch.cat([past_mask, causal_4d], dim=-1)
                    
                # 3. 擴充為 4D 形狀 [batch, 1, q_len, total_len]
                causal_4d = causal_4d[None, None, :, :].expand(bsz, 1, -1, -1).clone()
                
                # 4. 加上原始的 Padding Mask (確保 pad token 也被遮蔽)
                padding_4d = (1.0 - current_attention_mask[:, None, None, :]) * torch.finfo(self.model.dtype).min
                final_attention_mask = causal_4d + padding_4d
                # ====================================================================

                with torch.no_grad():
                    output_question_aware = self.model(
                        input_ids=current_ids,
                        attention_mask=final_attention_mask,  # <--- 這裡改成使用我們剛建構的 4D Mask
                        position_ids=position_ids,
                        output_attentions=True,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                current_seq_length = segment_context_ids.size(1)
                k = int(current_seq_length // self.compression_factor) + past_cache_len
                for layer_idx, layer_attention in enumerate(output_question_aware.attentions):
                    if self.mode == "attention_score":
                        summed_attention = layer_attention.sum(dim=1)
                        tot_seq_len = summed_attention.size(2)
                        if self.condition == "question":
                            context_attention = summed_attention[:, current_seq_length:, :current_seq_length + past_cache_len]
                            non_zero_counts = torch.arange(1, tot_seq_len + 1, device=context_attention.device)
                            non_zero_counts = non_zero_counts[current_seq_length + past_cache_len:]
                        elif self.condition == "context":
                            context_attention = summed_attention[:, :current_seq_length, :current_seq_length + past_cache_len]
                            non_zero_counts = torch.arange(1, tot_seq_len + 1, device=context_attention.device)
                            non_zero_counts = non_zero_counts[past_cache_len:past_cache_len + current_seq_length]
                        elif self.condition == "all":
                            context_attention = summed_attention[:, :, :current_seq_length + past_cache_len]
                            non_zero_counts = torch.arange(1, tot_seq_len + 1, device=context_attention.device)
                            non_zero_counts = non_zero_counts[past_cache_len:]
                        if self.normalize:
                            normalization_factors = non_zero_counts.float() / tot_seq_len
                            context_attention = context_attention * normalization_factors[None, :, None]
                        context_attention = summed_attention[:, current_seq_length:, :current_seq_length + past_cache_len]
                        aggregated_attention = context_attention.sum(dim=1)
                        _, important_tokens = torch.topk(aggregated_attention, k=k, dim=-1, largest=True)
                    elif self.mode == "cosine_similarity":
                        context_layer_embeddings = output_question_aware.hidden_states[layer_idx][:, :current_seq_length + past_cache_len]
                        question_layer_embedding = output_question_aware.hidden_states[layer_idx][:, current_seq_length:].mean(dim=1)
                        cosine_sim = torch.nn.functional.cosine_similarity(context_layer_embeddings,
                                                                            question_layer_embedding.unsqueeze(1), dim=-1)
                        _, important_tokens = torch.topk(cosine_sim, k=k, dim=-1, largest=True)
                    elif self.mode == "knn":
                        context_layer_embeddings = output_question_aware.hidden_states[layer_idx][:, :current_seq_length + past_cache_len]
                        question_layer_embedding = output_question_aware.hidden_states[layer_idx][:, current_seq_length:].mean(dim=1)
                        distances = torch.cdist(context_layer_embeddings.to(dtype=torch.double),
                                                question_layer_embedding.unsqueeze(1).to(dtype=torch.double), p=self.p).squeeze(
                            -1)
                        _, important_tokens = torch.topk(distances, k=k, dim=-1, largest=False)


                    elif self.mode == "svd":
                        context_layer_embeddings = output_question_aware.hidden_states[layer_idx][:, :current_seq_length + past_cache_len]
                        question_layer_embedding = output_question_aware.hidden_states[layer_idx][:, current_seq_length:].mean(dim=1)
                        important_tokens = torch.empty((context_layer_embeddings.shape[0], k),
                                                        device=context_layer_embeddings.device, dtype=torch.long)
                        for batch_idx, batch_context_layer_embeddings in enumerate(context_layer_embeddings):
                            # Add question embedding to each context embedding and perform PCA
                            combined_embeddings = batch_context_layer_embeddings + question_layer_embedding[batch_idx]
                            u, s, v = torch.pca_lowrank(combined_embeddings.to(dtype=torch.double), center=True, q=k + 2)

                            # Take the absolute values of the first column of u, sort and select top k
                            _, indices = torch.abs(u[:, 0]).sort(descending=True)
                            important_tokens[batch_idx] = indices[:k]
                    important_tokens, _ = torch.sort(important_tokens, dim=-1, descending=False)
                    past_key_values.update_rope(layer_idx, past_key_values.key_cache[layer_idx][:, :, :current_seq_length + past_cache_len], important_tokens)
                    # past_key_values.update_rope(layer_idx, past_key_values[layer_idx][0][:, :, :current_seq_length + past_cache_len], important_tokens)

                past_attention_mask = torch.ones(segment_attention_mask.size(0), k, device=segment_attention_mask.device, dtype=segment_attention_mask.dtype)
            end_processing_time = time.time()
            accelerator.log({
                    "target_token_mean": past_key_values.seen_tokens,
                    "target_token_min": past_key_values.seen_tokens,
                    "target_token_max": past_key_values.seen_tokens
            })
            start_generation_time = time.time()
            generate_kwargs['attention_mask'] = torch.cat([past_attention_mask, attention_mask], dim=-1)

            keys_to_remove = ["split_index", "context_ids", "context_attention_mask", "question_ids", "question_attention_mask"]
            for key in keys_to_remove:
                generate_kwargs.pop(key, None)

            model_output = super().generate(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, **generate_kwargs)
            end_generation_time = time.time()
            accelerator.log({"processing_time": end_processing_time - start_processing_time,
                        "generation_time": end_generation_time - start_generation_time}
                        )
            return model_output
        else:
            if self.target_token == 4096 and not self.is_full:
                context_ids_len = context_ids.size(1)
                context_ids = context_ids[:, :context_ids_len-input_ids.size(-1)]
                print("New context id size", context_ids.size(1))
                context_attention_mask = context_attention_mask[:, :context_ids_len-input_ids.size(-1)]
            start_processing_time = time.time()
            
            with torch.no_grad():
                self.model(
                    input_ids=context_ids,
                    attention_mask=context_attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values
                )
            end_processing_time = time.time()
            accelerator.log({
                "target_token_mean": past_key_values.seen_tokens,
                "target_token_min": past_key_values.seen_tokens,
                "target_token_max": past_key_values.seen_tokens
            })
            start_generation_time = time.time()
            generate_kwargs['attention_mask'] = torch.cat([context_attention_mask, attention_mask], dim=-1)
            
            keys_to_remove = ["split_index", "context_ids", "context_attention_mask", "question_ids", "question_attention_mask"]
            for key in keys_to_remove:
                generate_kwargs.pop(key, None)
                
            past_length = past_key_values.get_seq_length()
            dummy_input_ids = torch.zeros((input_ids.shape[0], past_length), dtype=input_ids.dtype, device=input_ids.device)
            full_input_ids = torch.cat([dummy_input_ids, input_ids], dim=1)
            
            model_output = super().generate(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, **generate_kwargs) # [:, context_ids_len:, ...]
            end_generation_time = time.time()
            accelerator.log({"processing_time": end_processing_time - start_processing_time,
                        "generation_time": end_generation_time - start_generation_time}
                        )
            return model_output



import math
import time
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from accelerate.logging import get_logger
from .base_qa_predictor import ModelQAPredictor
from ..metrics.longbench_metrics import compute_longbench_metric

logger = get_logger(__name__)

class LanguageModelingQAPredictor(ModelQAPredictor):
    def __init__(self, predictor_config, tokenizer, eval_examples, eval_dataset, data_config):
        super().__init__(predictor_config, tokenizer, eval_examples, eval_dataset, data_config)

    def post_processing_fn(self, predictions_data):
        # 修正 KeyError: 這裡改為根據實際預測到的 id 來建立回傳字典
        pred = {}
        for ex_id, text in predictions_data.items():
            pred[ex_id] = text
        return pred

    def predict(self, accelerator, model, dataloader):
        eos_token_ids = [self.tokenizer.eos_token_id]
        eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_token_id is not None and eot_token_id != self.tokenizer.unk_token_id:
            eos_token_ids.append(eot_token_id)
            
        gen_kwargs = {
            "max_new_tokens": self.predictor_config.max_answer_length,
            "num_beams": self.predictor_config.num_beams,
            "do_sample": self.predictor_config.do_sample,
            "temperature": self.predictor_config.temperature,
            "top_k": self.predictor_config.top_k,
            "top_p": self.predictor_config.top_p,
            "repetition_penalty": self.predictor_config.repetition_penalty,
            "eos_token_id": eos_token_ids,     
            "output_scores": True,               
            "return_dict_in_generate": True
        }
        
        model.eval()
        all_preds_for_metric = {}
        example_id_to_ref = {ex[self.data_config.id_column]: ex for ex in self.eval_examples}

        dataset_index = 0 
        aggregated_results = {}
        
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
            batch_size = batch["input_ids"].size(0)
            
            batch_dataset_slice = self.eval_dataset[dataset_index : dataset_index + batch_size]
            sample_ids = batch_dataset_slice.get("example_id", [f"unknown_{i}" for i in range(dataset_index, dataset_index + batch_size)])
            
            dataset_index += batch_size
            total_input_token = batch["input_ids"].size(1)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            with torch.no_grad():
                if "context_ids" in batch:
                    # New dataset: context_ids and question_ids already separate in batch
                    total_input_token = batch["context_ids"].size(1)
                    # Extract input portion (non-label tokens) for CLM generation
                    if "labels" in batch:
                        # Use per-example label mask (take first example's mask — all examples
                        # in a batch share the same prompt structure since pad_to_max_length=False
                        # and batch_size=1)
                        mask = batch["labels"][0] == -100
                        input_ids_for_generation = batch["input_ids"][:, mask]
                        input_ids_len = input_ids_for_generation.size(1)
                        attention_mask_for_generation = torch.ones(
                            (batch["input_ids"].size(0), input_ids_for_generation.size(1)),
                            dtype=torch.long, device=batch["attention_mask"].device
                        )
                        batch["input_ids"] = input_ids_for_generation
                        batch["attention_mask"] = attention_mask_for_generation
                        del batch["labels"]
                    else:
                        input_ids_len = batch["input_ids"].size(1)
                elif "split_index" in batch:
                    # Old dataset: split combined input_ids into context/question
                    split_idx = batch["split_index"][0].item()
                    input_ids_len = batch["input_ids"][:, split_idx:].size(1)
                    batch["context_ids"] = batch["input_ids"][:, :split_idx]
                    batch["context_attention_mask"] = batch["attention_mask"][:, :split_idx]
                    batch["question_ids"] = batch["input_ids"][:, split_idx:]
                    batch["question_attention_mask"] = batch["attention_mask"][:, split_idx:]
                    batch["input_ids"] = batch["question_ids"]
                    batch["attention_mask"] = batch["question_attention_mask"]
                else:
                    input_ids_len = batch["input_ids"].size(1)

                # 🌟 加入 try-except 區塊 🌟
                try:
                    model_unwrapped = accelerator.unwrap_model(model)
                    outputs = model_unwrapped.generate(
                        accelerator=accelerator,
                        **batch,
                        **gen_kwargs
                    )

                    e2e_latency_ms = (time.time() - start_time) * 1000
                    prefill_latency_ms = getattr(model_unwrapped, 'last_prefill_ms', 0.0)
                    decode_latency_ms = getattr(model_unwrapped, 'last_decode_ms', 0.0)
                    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

                    generated_ids = outputs.sequences
                    
                    transition_scores = accelerator.unwrap_model(model).compute_transition_scores(
                        generated_ids, outputs.scores, normalize_logits=True
                    )

                    num_new_tokens = len(outputs.scores)
                    if num_new_tokens > 0:
                        generated_ids_answer = generated_ids[:, -num_new_tokens:]
                    else:
                        generated_ids_answer = generated_ids
                        
                    decoded_text = self.tokenizer.batch_decode(generated_ids_answer, skip_special_tokens=True)

                    for i, text in enumerate(decoded_text):
                        s_id = sample_ids[i] if isinstance(sample_ids, list) else sample_ids
                        clean_text = text.replace("assistant\n", "").strip()
                        
                        valid_scores = transition_scores[i][transition_scores[i] != -np.inf]
                        avg_score = valid_scores.mean().item() if len(valid_scores) > 0 else -1e9
                        
                        if s_id not in aggregated_results:
                            aggregated_results[s_id] = {
                                "best_score": -1e9,
                                "output": "",
                                "e2e_latency_ms": 0.0,
                                "prefill_latency_ms": 0.0,
                                "decode_latency_ms": 0.0,
                                "peak_memory_mb": 0.0,
                                "input_token": total_input_token
                            }

                        aggregated_results[s_id]["e2e_latency_ms"] += e2e_latency_ms
                        aggregated_results[s_id]["prefill_latency_ms"] += prefill_latency_ms
                        aggregated_results[s_id]["decode_latency_ms"] += decode_latency_ms
                        aggregated_results[s_id]["peak_memory_mb"] = max(aggregated_results[s_id]["peak_memory_mb"], peak_mem)
                        
                        if avg_score > aggregated_results[s_id]["best_score"]:
                            aggregated_results[s_id]["best_score"] = avg_score
                            aggregated_results[s_id]["output"] = clean_text

                # 🌟 捕捉 OOM 錯誤 🌟
                except torch.cuda.OutOfMemoryError as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 強制釋放記憶體
                    logger.warning(f"⚠️ OOM 發生於 step {step} (ID: {sample_ids}, Tokens: {total_input_token})。略過此 Batch。")
                    
                    # 寫入預設的失敗紀錄
                    for i in range(batch_size):
                        s_id = sample_ids[i] if isinstance(sample_ids, list) else sample_ids
                        if s_id not in aggregated_results:
                            aggregated_results[s_id] = {
                                "best_score": -1e9,
                                "output": "[OOM_ERROR]",
                                "e2e_latency_ms": (time.time() - start_time) * 1000,
                                "prefill_latency_ms": 0.0,
                                "decode_latency_ms": 0.0,
                                "peak_memory_mb": 0.0,
                                "input_token": total_input_token
                            }
                except RuntimeError as e:
                    # 相容較舊版本 PyTorch 把 OOM 當作一般 RuntimeError 丟出的情況
                    if "out of memory" in str(e).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.warning(f"⚠️ OOM (RuntimeError) 發生於 step {step}。略過此 Batch。")
                        
                        for i in range(batch_size):
                            s_id = sample_ids[i] if isinstance(sample_ids, list) else sample_ids
                            if s_id not in aggregated_results:
                                aggregated_results[s_id] = {
                                    "best_score": -1e9,
                                    "output": "[OOM_ERROR]",
                                    "e2e_latency_ms": (time.time() - start_time) * 1000,
                                    "prefill_latency_ms": 0.0,
                                    "decode_latency_ms": 0.0,
                                    "peak_memory_mb": 0.0,
                                    "input_token": total_input_token
                                }
                    else:
                        raise e # 如果是其他錯誤，照常中斷報錯

        # --- 迴圈結束，將聚合結果轉換為 details_list 並結算分數 ---
        details_list = []
        final_preds, final_refs = [], []
        
        for s_id, data in aggregated_results.items():
            final_output = data["output"]
            all_preds_for_metric[s_id] = final_output
            
            ref_ex = example_id_to_ref.get(s_id, {})
            answer_col = getattr(self.data_config, "answer_column", "answers")
            ground_truth = ref_ex.get(answer_col, "")[0]
            
            sample_score = 0.0
            if ref_ex:
                sample_score = compute_longbench_metric(self.predictor_config.metric_name, [final_output], [ref_ex])
                final_preds.append(final_output)
                final_refs.append(ref_ex)
            
            details_list.append({
                "id": len(details_list),
                "input_token": data["input_token"],
                "output": final_output,
                "ground_truth": ground_truth,
                "f1_score": round(sample_score, 2),
                "e2e_latency_ms": round(data["e2e_latency_ms"], 2),
                "prefill_latency_ms": round(data["prefill_latency_ms"], 2),
                "decode_latency_ms": round(data["decode_latency_ms"], 2),
                "peak_memory_mb": round(data["peak_memory_mb"], 2)
            })

        overall_metric = compute_longbench_metric(self.predictor_config.metric_name, final_preds, final_refs) if final_preds else 0.0

        # --- 組裝 JSON 結構 ---
        task_name = getattr(self.data_config, "dataset_name", None) or str(self.data_config.test_file).split('/')[-1].split('.')[0]
        timestamp = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        os.makedirs("./logs", exist_ok=True)
        json_path = f"./logs/{timestamp}_{task_name}.json"

        model_unwrap = accelerator.unwrap_model(model)
        
        output_data = {
            "configs": {
                "task_name": task_name,
                "target_token": getattr(model_unwrap, 'target_token', "N/A"),
                "split_size": getattr(model_unwrap, 'split_size', "N/A"),
                "condition": getattr(model_unwrap, 'condition', "N/A"),
                "normalize": getattr(model_unwrap, 'normalize', "N/A"),
                "max_length": self.data_config.max_length,
                "context_max_length": getattr(self.data_config, 'context_max_length', "N/A"),
                "num_samples_evaluated": len(details_list),
                "max_answer_length": self.predictor_config.max_answer_length,
                "temperature": self.predictor_config.temperature
            },
            "result": {
                "avg_f1_score": round(overall_metric, 4),
                "avg_e2e_latency_ms": round(sum(d["e2e_latency_ms"] for d in details_list) / len(details_list), 2) if details_list else 0,
                "avg_prefill_latency_ms": round(sum(d["prefill_latency_ms"] for d in details_list) / len(details_list), 2) if details_list else 0,
                "avg_decode_latency_ms": round(sum(d["decode_latency_ms"] for d in details_list) / len(details_list), 2) if details_list else 0,
                "max_peak_memory": round(max(d["peak_memory_mb"] for d in details_list), 2) if details_list else 0
            },
            "details": details_list
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Detailed metrics saved to : {json_path}")

        return {self.metric_name: overall_metric}
    



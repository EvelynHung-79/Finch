import math
import time
import os
import json
import torch
import numpy as np
from tqdm import tqdm
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
        # 初始化性能追蹤指標
        if accelerator.is_main_process:
            # 這裡保留原有的 wandb 設定
            pass

        gen_kwargs = {
            "max_new_tokens": self.predictor_config.max_answer_length,
            "num_beams": self.predictor_config.num_beams,
            "do_sample": self.predictor_config.do_sample,
            "temperature": self.predictor_config.temperature,
            "top_k": self.predictor_config.top_k,
            "top_p": self.predictor_config.top_p,
            "repetition_penalty": self.predictor_config.repetition_penalty
        }
        
        model.eval()
        per_sample_details = {} # 用來存 Json 的詳細資料
        all_preds_for_metric = {} # 用來傳給評分函式的文字結果

        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
            # 取得 Sample ID (LongBench 預設欄位通常是 _id 或 id)
            # 從 batch 中提取，若無則使用順序編號
            sample_ids = batch.get("_id", batch.get("id", [f"unknown_{step}"] * batch["input_ids"].size(0)))

            # 開始測量效能
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            with torch.no_grad():
                # 處理輸入長度邏輯 (用於切掉 Prompt)
                if "split_index" in batch:
                    split_idx = batch["split_index"][0].item()
                    input_ids_len = batch["input_ids"][:, split_idx:].size(1)
                    # 重新打包 batch 給 generate
                    batch["context_ids"] = batch["input_ids"][:, :split_idx]
                    batch["context_attention_mask"] = batch["attention_mask"][:, :split_idx]
                    batch["question_ids"] = batch["input_ids"][:, split_idx:]
                    batch["question_attention_mask"] = batch["attention_mask"][:, split_idx:]
                    batch["input_ids"] = batch["question_ids"]
                    batch["attention_mask"] = batch["question_attention_mask"]
                else:
                    input_ids_len = batch["input_ids"].size(1)

                # 執行生成
                generated_ids = accelerator.unwrap_model(model).generate(
                    accelerator=accelerator,
                    **batch,
                    **gen_kwargs
                )
                
                # 結束測量
                end_time = time.time()
                latency = end_time - start_time
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

                # 解碼文字
                if generated_ids.size(1) > input_ids_len:
                    generated_ids = generated_ids[:, input_ids_len:]
                
                decoded_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # 紀錄到字典中
                for i, text in enumerate(decoded_text):
                    s_id = sample_ids[i] if isinstance(sample_ids, list) else sample_ids
                    clean_text = text.strip()
                    per_sample_details[s_id] = {
                        "output": clean_text,
                        "latency_sec": round(latency, 4),
                        "peak_memory_mb": round(peak_mem, 2)
                    }
                    all_preds_for_metric[s_id] = clean_text

        # 儲存詳細結果到 JSON
        if self.predictor_config.output_file_path:
            json_path = self.predictor_config.output_file_path.replace(".csv", ".json")
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(per_sample_details, f, indent=4, ensure_ascii=False)
            logger.info(f"Detailed metrics saved to {json_path}")

        # 進行評分
        # 從 eval_examples 抓取對應的解答 (references)
        example_id_to_ref = {ex[self.data_config.id_column]: ex for ex in self.eval_examples}
        
        final_preds = []
        final_refs = []
        for s_id, text in all_preds_for_metric.items():
            if s_id in example_id_to_ref:
                final_preds.append(text)
                final_refs.append(example_id_to_ref[s_id])

        if final_preds:
            predict_metric = compute_longbench_metric(self.predictor_config.metric_name, final_preds, final_refs)
        else:
            predict_metric = 0.0

        return {self.metric_name: predict_metric}
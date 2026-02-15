import math
from abc import ABC
from typing import Tuple

import torch
import wandb
from accelerate.logging import get_logger
from tqdm import tqdm
import numpy as np

from ..logging_utils import log_wandb_table, log_predictions_as_csv
from .base_qa_predictor import ModelQAPredictor
from ..metrics.longbench_metrics import compute_longbench_metric

logger = get_logger(__name__)


class LanguageModelingQAPredictor(ModelQAPredictor, ABC):
    def __init__(self, predictor_config, tokenizer, eval_examples, eval_dataset, data_config):
        super().__init__(predictor_config, tokenizer, eval_examples, eval_dataset, data_config)


    def post_processing_fn(
        self,
        predictions: Tuple[np.ndarray, np.ndarray]
    ):
        example_id_to_index = {k: i for i, k in enumerate(self.eval_examples[self.data_config.id_column])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(self.eval_dataset)}
        pred = {}
        for example_index, example in enumerate(tqdm(self.eval_examples)):
            # This is the index of the feature associated to the current example.
            feature_index = feature_per_example[example_index]
            pred[example[self.data_config.id_column]] = predictions[feature_index]
        return pred

    def predict(self, accelerator, model, dataloader):

        if accelerator.is_main_process:
            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
            wandb_tracker.define_metric("processing_time", summary="mean")
            wandb_tracker.define_metric("generation_time", summary="mean")
            wandb_tracker.define_metric("context_size_mean", summary="mean")
            wandb_tracker.define_metric("context_size_min", summary="min")
            wandb_tracker.define_metric("context_size_max", summary="max")
            wandb_tracker.define_metric("target_token_mean", summary="mean")
            wandb_tracker.define_metric("target_token_min", summary="min")
            wandb_tracker.define_metric("target_token_max", summary="max")
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
        losses = []
        predictions = []
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
            with torch.no_grad():
                if self.compute_perplexity:
                    outputs = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
                    loss = outputs.loss
                    losses.append(
                        accelerator.gather_for_metrics(loss.repeat(self.predictor_config.batch_size)))
                if "split_index" in batch:
                    # 取得分割點 (假設 batch_size=1，或是 batch 內長度一致)
                    # 注意：DataCollator 應該會把 split_index 轉成 Tensor
                    split_idx = batch["split_index"][0].item()
                    
                    full_input_ids = batch["input_ids"]
                    full_attention_mask = batch["attention_mask"]

                    # 根據 Tokenizer 的 padding_side 邏輯 (Llama 預設 Left Padding)
                    # 序列結構通常是: [PAD, PAD, Context(0), Question(1)]
                    # split_idx 指向 Question(1) 的開頭
                    
                    # 切割出 Context (包含前面的 Padding)
                    context_ids = full_input_ids[:, :split_idx]
                    context_attention_mask = full_attention_mask[:, :split_idx]
                    
                    # 切割出 Question (作為 Prompt)
                    question_ids = full_input_ids[:, split_idx:]
                    question_attention_mask = full_attention_mask[:, split_idx:]
                    
                    # 更新 batch，傳入 model.generate 所需的參數
                    batch["context_ids"] = context_ids
                    batch["context_attention_mask"] = context_attention_mask
                    
                    batch["question_ids"] = question_ids
                    batch["question_attention_mask"] = question_attention_mask
                    
                    # 更新 input_ids 為 Question (讓模型接續生成)
                    batch["input_ids"] = question_ids
                    batch["attention_mask"] = question_attention_mask
                    
                    # 設定 input_ids_len，讓後續程式碼知道要切掉 Prompt
                    input_ids_len = question_ids.size(1)

                elif self.predictor_config.task_type == 'CLM':
                    # 舊的 fallback 邏輯 (如果 dataset 沒更新會跑這，但現在應該用不到)
                    if "labels" in batch:
                         input_ids_for_generation = batch["input_ids"][:, batch["labels"][0] == -100]
                    else:
                         input_ids_for_generation = batch["input_ids"]
                    input_ids_len = input_ids_for_generation.size(1)
                generated_ids = accelerator.unwrap_model(model).generate(
                    accelerator=accelerator,
                    **batch,
                    **gen_kwargs
                )
                if self.predictor_config.task_type == 'CLM' and generated_ids.size(1) > input_ids_len:
                    generated_ids = generated_ids[:, input_ids_len:, ...]
                generated_ids = accelerator.pad_across_processes(
                    generated_ids, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                generated_ids = accelerator.gather(generated_ids).cpu().numpy()
                predictions.append(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        predictions_concat = np.concatenate(predictions, axis=0)
        predictions, references = self.post_processing(predictions_concat)
        # log_wandb_table(accelerator, predictions, references, self.data_config)
        # log_predictions_as_csv(predictions, references, self.predictor_config.output_file_path, self.data_config)
        #references = [
        #    {k: v for k, v in ex.items() if k not in [self.data_config.question_column, self.data_config.context_column]} for ex in references
        #]
        if "squad" in self.metric_name:
            predict_metric = self.metric.compute(predictions=predictions, references=references)
        elif "rouge" in self.metric_name:
            predict_metric = self.metric.compute(predictions=predictions, references=references, use_stemmer=True)
        else:
            predict_metric = compute_longbench_metric(self.predictor_config.metric_name, predictions, references)
        results = {
            self.metric_name: predict_metric,
        }

        if self.compute_perplexity:
            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                eval_loss = torch.mean(losses)
                perplexity = float("inf")
            results["perplexity"] = perplexity
            results["eval_loss"] = eval_loss

        return results

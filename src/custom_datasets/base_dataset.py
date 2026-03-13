from abc import ABC, abstractmethod

from datasets import load_dataset
from transformers import PreTrainedTokenizer, default_data_collator, DataCollatorWithPadding


class BaseDataset(ABC):

    def __init__(self, tokenizer: PreTrainedTokenizer, model, split: str, data_config):
        self.data_config = data_config
        self.column_names = None
        self.split = split
        self.tokenizer = tokenizer
        self.model = model
        self.columns_to_remove_for_model = []

    def filter(self, example):
        return True

    def load(self):
        # Load the dataset
        if self.data_config.dataset_name:
            print(f"Loading dataset {self.data_config.dataset_name} with config {self.data_config.dataset_config_name}")
            raw_datasets = load_dataset(self.data_config.dataset_name, self.data_config.dataset_config_name)
            print("Loaded!")
        else:
            data_files = {}
            if self.data_config.train_file:
                data_files["train"] = self.data_config.train_file
            if self.data_config.validation_file:
                data_files["validation"] = self.data_config.validation_file
            if self.data_config.test_file:
                data_files["test"] = self.data_config.test_file
                extension = "json"
            raw_datasets = load_dataset(extension, data_files=data_files)
        self.column_names = raw_datasets[self.split].column_names

        max_samples = getattr(self.data_config, "max_eval_samples", None)
        
        if max_samples is not None:
            total_len = len(raw_datasets[self.split])
            actual_samples = min(max_samples, total_len)
            raw_datasets[self.split] = raw_datasets[self.split].select(range(actual_samples))
            print(f"✅ 成功在切塊前，將原始資料限制為 {actual_samples} 篇獨立文章 (Samples)")
            
        return raw_datasets[self.split]

    def get_data_collator(self):
        if self.data_config.pad_to_max_length:
            return default_data_collator
        else:
            return DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=(8 if self.data_config.use_fp16 else None))


    @abstractmethod
    def tokenize(self, examples):
        raise NotImplementedError(f"{self.__class__.__name__} must implement the tokenize method.")

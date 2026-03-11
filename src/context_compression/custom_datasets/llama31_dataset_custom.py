from abc import ABC

from transformers import PreTrainedTokenizer, default_data_collator

from .base_dataset import BaseDataset

from accelerate.logging import get_logger
logger = get_logger(__name__)


class Llama31DatasetCustom(BaseDataset, ABC):
    """
    Dataset class for Llama 3.1 Instruct that tokenizes context and question SEPARATELY,
    matching the original FINCH paper design (LlamaDatasetCustom for Llama 2).

    Context is tokenized independently (up to context_max_length tokens) so FINCH can
    process the full document iteratively via split_size chunking in the model's generate().
    """

    def __init__(self, split: str, data_config, tokenizer: PreTrainedTokenizer, model):
        super().__init__(tokenizer, model, split, data_config)
        self.pad_to_max_length = data_config.pad_to_max_length
        self.max_seq_length = data_config.max_length
        self.max_answer_length = data_config.max_answer_length
        self.max_question_length = data_config.question_max_length
        self.max_context_length = data_config.context_max_length
        self.columns_to_remove_for_model = ["example_id"]
        self.question_column = data_config.question_column
        self.context_column = data_config.context_column
        self.answer_column = data_config.answer_column
        self.id_column = data_config.id_column
        self.system_prompt = data_config.prompt
        self.question_prompt = getattr(data_config, 'question_prompt', None)

        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        self.tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    def generate_input(self, _question):
        """Format question portion for generation (comes AFTER context in KV cache).

        The full sequence seen by the model is:
          [context_ids] + [input_ids]
        = [<|begin_of_text|><|system|>prompt<|eot_id|><|user|>context: ...] + [question: ...<|eot_id|><|assistant|>]

        So input_ids only contains the question + closing user turn + assistant header.
        """
        return (
            f"\nquestion: {_question.lstrip()}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    @staticmethod
    def generate_question(_question):
        return f"question: {_question.lstrip()}"

    @staticmethod
    def generate_context(_context, system_prompt):
        """Format context with chat headers so the full sequence is valid Llama 3.1 Instruct format.

        context_ids will contain: <|begin_of_text|><|system|>prompt<|eot_id|><|user|>context: {text}
        When concatenated with input_ids: ...question: {q}<|eot_id|><|assistant|>
        The result is a well-formed Llama 3.1 Instruct conversation.
        """
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"context: {_context.lstrip()}"
        )

    @staticmethod
    def extract_targets(answers):
        if all(isinstance(answer, dict) and 'text' in answer for answer in answers):
            return [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
        elif all(isinstance(answer, list) and len(answer) > 0 and isinstance(answer[0], str) for answer in answers):
            return [answer[0] for answer in answers]
        else:
            raise ValueError("The structure of the answers field is not recognized.")

    def tokenize(self, examples):
        if self.question_prompt is not None:
            questions = [self.question_prompt] * len(examples[self.context_column])
        else:
            questions = examples[self.question_column]

        contexts = examples[self.context_column]
        answers = examples[self.answer_column]
        targets = self.extract_targets(answers)

        # 1. Tokenize the full chat-formatted input (system + question + assistant header)
        inputs = [self.generate_input(q) for q in questions]
        max_length = self.max_seq_length + self.max_answer_length
        tokenized_examples = self.tokenizer(
            inputs,
            targets,
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation="only_first"
        )

        # 2. Tokenize question separately (for FINCH's prompt-guided compression)
        if self.question_prompt is not None:
            question_texts = questions  # already [question_prompt] * n from above
        else:
            question_texts = [self.generate_question(q) for q in questions]
        tokenized_questions = self.tokenizer(
            question_texts,
            add_special_tokens=False,
            max_length=self.max_question_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True
        )

        # 3. Tokenize context separately (full document, up to context_max_length)
        #    Context includes chat headers so the full sequence is valid Llama 3.1 format
        context_texts = [self.generate_context(c, self.system_prompt) for c in contexts]
        tokenized_contexts = self.tokenizer(
            context_texts,
            add_special_tokens=False,
            max_length=self.max_context_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True
        )

        # Add EOS token to input_ids
        labels = self.tokenizer(targets, add_special_tokens=False)
        for idx, input_ids in enumerate(tokenized_examples["input_ids"]):
            tokenized_examples["input_ids"][idx] = input_ids + [self.tokenizer.eos_token_id]
            tokenized_examples["attention_mask"][idx] = tokenized_examples["attention_mask"][idx] + [1]

        # Build labels: -100 for non-answer tokens, actual ids for answer tokens
        for idx, input_ids in enumerate(tokenized_examples["input_ids"]):
            label_input_ids = labels["input_ids"][idx] + [self.tokenizer.eos_token_id]
            labels["input_ids"][idx] = [-100] * (len(input_ids) - len(label_input_ids)) + label_input_ids

        if self.split == "train":
            tokenized_examples["labels"] = labels["input_ids"]
        else:
            tokenized_examples["example_id"] = []
            labels_out = []
            for i in range(len(tokenized_examples["input_ids"])):
                tokenized_examples["example_id"].append(examples[self.id_column][i])
                labels_out.append(labels["input_ids"][i])
            tokenized_examples["labels"] = labels_out

        # Attach separately-tokenized context and question
        tokenized_examples["question_ids"] = tokenized_questions["input_ids"]
        tokenized_examples["question_attention_mask"] = tokenized_questions["attention_mask"]
        tokenized_examples["context_ids"] = tokenized_contexts["input_ids"]
        tokenized_examples["context_attention_mask"] = tokenized_contexts["attention_mask"]

        return tokenized_examples

    def get_data_collator(self):
        return default_data_collator

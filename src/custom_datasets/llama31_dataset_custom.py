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

    Prompt modes
    ------------
    Template mode (recommended):
        Set ``prompt`` in the yaml to a string containing ``{context}`` and optionally
        ``{input}`` placeholders, e.g.::

            "Answer the question.\n\n{context}\n\nQuestion: {input}\nAnswer:"

        The string is split at ``{context}`` and ``{input}`` into three parts:

        * ``before_context``  – prepended before the context document in context_ids
        * ``ctx_input_sep``   – inserted between context and question in context_ids
        * ``after_input``     – appended after the question in input_ids (before <eot>)

        The full token sequence seen by the model is:
          [<bos><user> before_context + context + ctx_input_sep] + [input + after_input <eot><assistant>]

        ``condition=question`` still works normally: the raw ``input`` field (or
        ``question_prompt`` if set) is used as the FINCH compression condition,
        independent of the prompt template.

    Legacy mode:
        If ``{context}`` is NOT present in ``prompt``, the original behaviour is used:
          [<bos><system>prompt<eot><user>context: {context}] + [\nquestion: {input}<eot><assistant>]
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

        # Detect template mode: prompt must contain {context}
        self.use_template = "{context}" in self.system_prompt
        if self.use_template:
            ctx_idx = self.system_prompt.index("{context}")
            self._before_context = self.system_prompt[:ctx_idx]
            after_context = self.system_prompt[ctx_idx + len("{context}"):]
            if "{input}" in after_context:
                inp_idx = after_context.index("{input}")
                self._ctx_input_sep = after_context[:inp_idx]
                self._after_input = after_context[inp_idx + len("{input}"):]
            else:
                self._ctx_input_sep = after_context
                self._after_input = ""

        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        self.tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    def generate_input(self, _question):
        """Format the question portion (appended after context_ids during generation)."""
        if self.use_template:
            return (
                f"{_question.lstrip()}{self._after_input}"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        return (
            f"\nquestion: {_question.lstrip()}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate_question(self, _question):
        """Return the text used as FINCH's compression condition (condition=question)."""
        if self.use_template:
            return _question.lstrip()
        return f"question: {_question.lstrip()}"

    def generate_context(self, _context):
        """Format the context portion (processed by FINCH chunk-by-chunk)."""
        if self.use_template:
            return (
                f"<|begin_of_text|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{self._before_context}{_context.lstrip()}{self._ctx_input_sep}"
            )
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
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
        raw_questions = examples[self.question_column]
        contexts = examples[self.context_column]
        answers = examples[self.answer_column]
        targets = self.extract_targets(answers)

        # 1. Tokenize the full chat-formatted input (question + assistant header).
        #    Always use the real question text from the dataset so the prompt template
        #    can substitute {input} correctly.  question_prompt only affects the
        #    FINCH compression condition (step 2), not the actual input sequence.
        inputs = [self.generate_input(q) for q in raw_questions]
        max_length = self.max_seq_length + self.max_answer_length
        tokenized_examples = self.tokenizer(
            inputs,
            targets,
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation="longest_first"
        )

        # 2. Tokenize question separately (for FINCH's prompt-guided compression).
        #    question_prompt overrides the condition signal when the raw input is
        #    empty or uninformative (e.g. gov_report, multi_news, passage_count, lcc).
        if self.question_prompt is not None:
            question_texts = [self.question_prompt] * len(raw_questions)
        else:
            question_texts = [self.generate_question(q) for q in raw_questions]
        tokenized_questions = self.tokenizer(
            question_texts,
            add_special_tokens=False,
            max_length=self.max_question_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True
        )

        # 3. Tokenize context separately (full document, up to context_max_length)
        #    Context includes chat headers so the full sequence is valid Llama 3.1 format
        context_texts = [self.generate_context(c) for c in contexts]
        tokenized_contexts = self.tokenizer(
            context_texts,
            add_special_tokens=False,
            max_length=self.max_context_length,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True
        )

        # Add EOS token to input_ids
        labels = self.tokenizer(
            targets,
            add_special_tokens=False,
            max_length=self.max_answer_length,
            truncation=True
        )
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

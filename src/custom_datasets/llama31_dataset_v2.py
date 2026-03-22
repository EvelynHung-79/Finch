from .llama31_dataset_custom import Llama31DatasetCustom
from accelerate.logging import get_logger

logger = get_logger(__name__)


class Llama31DatasetV2(Llama31DatasetCustom):
    """
    Dataset class for LongBench v2 multiple-choice tasks.

    LongBench v2 data format:
        - question: the question text
        - choice_A / choice_B / choice_C / choice_D: the four answer options
        - answer: single letter string, one of "A", "B", "C", "D"
        - context: the long document

    This class pre-processes the raw data so it is compatible with the base
    Llama31DatasetCustom pipeline:
        - Merges question + choices into a single ``input`` column so that
          {input} in the prompt template is substituted correctly.
        - Wraps the single-letter ``answer`` string into a one-element list
          stored under the ``answers`` column, matching the format that
          ``extract_targets`` and ``compute_longbench_metric`` expect.
    """

    def load(self):
        raw_split = super().load()

        def _preprocess(example):
            q = example.get("question", "")
            a = example.get("choice_A", "")
            b = example.get("choice_B", "")
            c = example.get("choice_C", "")
            d = example.get("choice_D", "")
            example["input"] = (
                f"Question:\n{q}\n\n"
                f"Options:\n"
                f"A. {a}\n"
                f"B. {b}\n"
                f"C. {c}\n"
                f"D. {d}"
            )
            example["answers"] = [example.get("answer", "")]
            return example

        processed = raw_split.map(_preprocess)
        self.column_names = processed.column_names
        return processed

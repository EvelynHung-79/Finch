import re
import string
from collections import Counter
from fuzzywuzzy import fuzz
from rouge_score import rouge_scorer

# 新增 1：正規化答案的輔助函數 (LongBench 標準作法)
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# 新增 2：計算字詞級別 F1 Score 的函數
def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_longbench_metric(type, predictions, references):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, references):
        max_score = 0.
        answers = ground_truths["answers"]
        for ground_truth in answers:
            if type == "classification_score":
                # prediction = prediction.lstrip('\n').split('\n')[0]
                all_classes = ground_truths["all_classes"]
                em_match_list = []
                for class_name in all_classes:
                    if class_name in prediction:
                        em_match_list.append(class_name)
                for match_term in em_match_list:
                    if match_term in ground_truth and match_term != ground_truth:
                        em_match_list.remove(match_term)
                if ground_truth in em_match_list:
                    score = (1.0 / len(em_match_list))
                else:
                    score = 0.0
                max_score = max(score, max_score)
            elif type == "longbench_qa":
                score = qa_f1_score(prediction, ground_truth)
                max_score = max(score, max_score)
            elif type == "code_sim_score":
                all_lines = prediction.lstrip('\n').split('\n')
                prediction = ""
                for line in all_lines:
                    if ('`' not in line) and ('#' not in line) and ('//' not in line):
                        prediction = line
                        break
                max_score = max(fuzz.ratio(prediction, ground_truth) / 100, max_score)
            elif type == "count_score":
                numbers = re.findall(r"\d+", prediction)
                right_num = 0
                for number in numbers:
                    if str(number) == str(ground_truth):
                        right_num += 1
                final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)

                max_score = max(final_score, max_score)
            elif type == "retrieval_score":
                pattern = r'Paragraph (\d+)'
                matches = re.findall(pattern, ground_truth)
                ground_truth_id = matches[0]
                numbers = re.findall(r"\d+", prediction)
                right_num = 0
                for number in numbers:
                    if str(number) == str(ground_truth_id):
                        right_num += 1
                final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
                max_score=max(float(final_score), max_score)
            elif type == "longbench_summarization":
                scores = _rouge_scorer.score(ground_truth, prediction)
                score = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
                max_score = max(score, max_score)
        total_score += max_score
    score = round(100 * total_score / len(predictions), 2)
    return score

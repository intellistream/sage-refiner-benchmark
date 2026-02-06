"""
LongBench 官方评估指标函数

直接照搬自: https://github.com/THUDM/LongBench/blob/main/metrics.py
添加了类型注解和可选依赖处理。
"""

import re
import string
from collections import Counter
from typing import List

# 可选依赖
try:
    import jieba

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

try:
    from fuzzywuzzy import fuzz

    HAS_FUZZYWUZZY = True
except ImportError:
    HAS_FUZZYWUZZY = False

try:
    from rouge import Rouge

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Lower text and remove punctuation, extra whitespace (Chinese)."""

    def white_space_fix(text: str) -> str:
        return "".join(text.split())

    def remove_punc(text: str) -> str:
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛"
            "„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction: List[str], ground_truth: List[str], **kwargs) -> float:
    """Token-level F1 score."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """QA F1 score (English)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """QA F1 score (Chinese, requires jieba)."""
    if not HAS_JIEBA:
        raise ImportError(
            "jieba is required for Chinese evaluation. Install with: pip install jieba"
        )

    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """ROUGE-L F1 score."""
    if not HAS_ROUGE:
        raise ImportError("rouge is required. Install with: pip install rouge")

    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        return scores["rouge-l"]["f"]
    except Exception:
        return 0.0


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """ROUGE-L F1 score (Chinese, requires jieba)."""
    if not HAS_JIEBA:
        raise ImportError(
            "jieba is required for Chinese evaluation. Install with: pip install jieba"
        )

    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Classification score with all_classes matching."""
    all_classes = kwargs.get("all_classes", [])
    if not all_classes:
        return 0.0

    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)

    for match_term in em_match_list.copy():
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)

    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Retrieval score (English)."""
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Retrieval score (Chinese)."""
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Count score for passage_count task."""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Code similarity score (requires fuzzywuzzy)."""
    if not HAS_FUZZYWUZZY:
        raise ImportError(
            "fuzzywuzzy is required for code evaluation. "
            "Install with: pip install fuzzywuzzy python-Levenshtein"
        )

    all_lines = prediction.lstrip("\n").split("\n")
    processed_prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            processed_prediction = line
            break
    return fuzz.ratio(processed_prediction, ground_truth) / 100.0


# 指标函数注册表（照搬官方 dataset2metric 映射）
METRIC_FUNCTIONS = {
    "qa_f1": qa_f1_score,
    "qa_f1_zh": qa_f1_zh_score,
    "rouge": rouge_score,
    "rouge_zh": rouge_zh_score,
    "classification": classification_score,
    "retrieval": retrieval_score,
    "retrieval_zh": retrieval_zh_score,
    "count": count_score,
    "code_sim": code_sim_score,
}

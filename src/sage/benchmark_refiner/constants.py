"""
LongBench 常量定义

数据集到评估指标的映射（照搬官方 eval.py）
"""

# 数据集到评估指标的映射（来自官方 eval.py）
DATASET_TO_METRIC: dict[str, str] = {
    # QA 任务 - F1 score
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "triviaqa": "qa_f1",
    # 中文 QA（需要 jieba 分词）
    "multifieldqa_zh": "qa_f1_zh",
    # 摘要任务 - ROUGE score
    "gov_report": "rouge",
    "qmsum": "rouge",
    "multi_news": "rouge",
    "samsum": "rouge",
    # 中文摘要（需要 jieba 分词）
    "dureader": "rouge_zh",
    "vcsum": "rouge_zh",
    # 分类任务
    "trec": "classification",
    "lsht": "classification",
    # 检索任务
    "passage_retrieval_en": "retrieval",
    "passage_retrieval_zh": "retrieval_zh",
    "passage_count": "count",
    # 代码任务（需要 fuzzywuzzy）
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}

# 需要取第一行的数据集（照搬官方 eval.py）
FIRST_LINE_DATASETS: set[str] = {"trec", "triviaqa", "samsum", "lsht"}

# 不使用 chat template 的数据集（照搬官方 pred.py 注释）
# chat models are better off without build prompts on these tasks
NO_CHAT_DATASETS: set[str] = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

# 支持的数据集列表
SUPPORTED_DATASETS: set[str] = set(DATASET_TO_METRIC.keys())

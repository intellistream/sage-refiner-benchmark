"""
Context Compression & Refiner Benchmark Module
===============================================

上下文压缩与 Refiner 算法性能评估组件。

支持的算法/数据集：
- LongBench (THUDM/LongBench) - 长文本理解基准
- LongRefiner - 上下文压缩算法
- REFORM - 检索优化
- Provence - 上下文剪枝
- 其他自定义 Refiner 实现

组件：
- LongBenchBatch: 从 THUDM/LongBench 加载数据
- LongBenchPromptor: 生成 LongBench 官方 prompt
- LongBenchEvaluator: LongBench 官方评估指标

使用示例：
    from sage.benchmark_refiner import (
        LongBenchBatch,
        LongBenchPromptor,
        LongBenchEvaluator,
    )

    # 在 pipeline 中使用
    env.from_batch(LongBenchBatch, config["source"])
       .map(LongBenchPromptor, config["promptor"])
       .map(OpenAIGenerator, config["generator"])
       .map(LongBenchEvaluator, config["evaluate"])

配置示例：
    source:
      hf_dataset_name: "THUDM/LongBench"
      hf_dataset_config: "hotpotqa"
      max_samples: 100

    promptor:
      max_input_tokens: 120000
      is_chat_model: true
      model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"

    evaluate:
      longbench_e_buckets: false
      output_path: "results/longbench_results.jsonl"
      model_name: "Qwen/Qwen2.5-7B-Instruct"

依赖安装：
    pip install isage-refiner-benchmark[refiner]

    # 可选依赖说明：
    # - jieba: 中文分词（中文数据集评估需要）
    # - fuzzywuzzy + python-Levenshtein: 代码相似度（lcc, repobench-p 需要）
    # - rouge: ROUGE-L 分数（摘要任务需要）
"""

from .batch import LongBenchBatch
from .evaluator import LongBenchEvaluator
from .promptor import LongBenchPromptor

__all__ = [
    "LongBenchBatch",
    "LongBenchPromptor",
    "LongBenchEvaluator",
]

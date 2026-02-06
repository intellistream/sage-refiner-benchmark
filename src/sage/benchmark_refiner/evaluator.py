"""
LongBench Evaluator - 优化版

改进点：
1. 模型专用后处理（照搬官方 post_process）
2. 结果持久化（JSONL 逐条写入）
3. 压缩文本追踪
4. 时间统计
"""

import json
from pathlib import Path
from typing import Any, Optional, TextIO

from sage.common.core import StopSignal
from sage.common.core.functions import MapFunction as MapOperator

from .constants import DATASET_TO_METRIC, FIRST_LINE_DATASETS
from .metrics import METRIC_FUNCTIONS
from .utils import post_process


class LongBenchEvaluator(MapOperator):
    """
    LongBench 专用评估器。

    功能：
    1. 根据数据集自动选择评估指标
    2. 支持标准版单一分数和 LongBench-E 长度分桶
    3. 集成所有 LongBench 指标函数
    4. 预测结果后处理（特定数据集只取第一行）
    5. 模型专用后处理（照搬官方 post_process）
    6. 结果持久化（JSONL 格式）

    输入数据格式（来自 Generator）：
    {
        "query": str,
        "generated": str,          # 模型生成的答案
        "references": List[str],   # 标准答案列表
        "_dataset": str,           # 数据集名称
        "all_classes": List[str],  # 分类任务类别（可选）
        "length": int,             # 原始长度（LongBench-E 分桶用）
    }

    配置参数：
        - longbench_e_buckets: bool - 是否输出 LongBench-E 分桶分数
        - output_path: str | None - 结果保存路径（JSONL 格式）
        - model_name: str | None - 模型名称（用于后处理）
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config or {}
        self.longbench_e_buckets: bool = self.config.get("longbench_e_buckets", False)
        self.model_name: str = self.config.get("model_name", "")

        # 结果持久化
        self._output_path: Optional[str] = self.config.get("output_path")
        self._output_file: Optional[TextIO] = None
        if self._output_path:
            Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
            self._output_file = open(self._output_path, "a", encoding="utf-8")

        # 分数收集器（用于计算平均分）
        self._scores: list[float] = []
        self._dataset_scores: dict[str, list[float]] = {}

        # LongBench-E 分桶分数
        self._bucket_scores: dict[str, list[float]] = {
            "0-4k": [],
            "4-8k": [],
            "8k+": [],
        }

        # 时间收集器
        self._refine_times: list[float] = []
        self._generate_times: list[float] = []
        self._retrieve_times: list[float] = []

    def _post_process_prediction(self, pred: str, dataset: str) -> str:
        """预测结果后处理"""
        # 1. 模型专用后处理（照搬官方）
        if self.model_name:
            pred = post_process(pred, self.model_name)

        # 2. 数据集专用后处理：取第一行（照搬官方 eval.py）
        if dataset in FIRST_LINE_DATASETS:
            pred = pred.lstrip("\n").split("\n")[0]

        return pred

    def _get_length_bucket(self, length: int) -> str:
        """根据长度获取分桶名称"""
        if length < 4000:
            return "0-4k"
        elif length < 8000:
            return "4-8k"
        return "8k+"

    def _compute_score(
        self,
        pred: str,
        ground_truths: list[str],
        dataset: str,
        all_classes: Optional[list[str]] = None,
    ) -> float:
        """计算单个样本的分数"""
        # 获取指标类型
        metric_type = DATASET_TO_METRIC.get(dataset, "qa_f1")
        metric_fn = METRIC_FUNCTIONS.get(metric_type)

        if metric_fn is None:
            self.logger.warning(f"Unknown metric type: {metric_type}")
            return 0.0

        # 后处理预测结果
        pred = self._post_process_prediction(pred, dataset)

        # 对所有参考答案计算分数，取最高（照搬官方 eval.py scorer）
        best_score = 0.0
        for ground_truth in ground_truths:
            try:
                score = metric_fn(pred, ground_truth, all_classes=all_classes or [])
                best_score = max(best_score, score)
            except Exception as e:
                self.logger.warning(f"Error computing score for {dataset}: {e}")

        return best_score

    def _save_result(self, data: dict[str, Any], score: float, dataset: str) -> None:
        """保存单条结果到 JSONL 文件"""
        if not self._output_file:
            return

        result: dict[str, Any] = {
            "pred": data.get("generated", ""),
            "answers": data.get("references", []),
            "score": score,
            "dataset": dataset,
            "all_classes": data.get("all_classes"),
            "length": data.get("length", 0),
        }

        # 保存压缩后的上下文（如果存在）
        refining_results = data.get("refining_results", [])
        if refining_results:
            if isinstance(refining_results, list):
                result["compressed_text"] = "\n\n".join(refining_results)
            else:
                result["compressed_text"] = str(refining_results)

        # 保存时间数据
        times: dict[str, float] = {}
        if "retrieve_time" in data:
            times["retrieve"] = data["retrieve_time"]
        if "refine_time" in data:
            times["refine"] = data["refine_time"]
        if "generate_time" in data:
            times["generate"] = data["generate_time"]
        if times:
            result["times"] = times

        self._output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        self._output_file.flush()

    def execute(self, data: Any) -> Any:
        """执行评估"""
        # Handle StopSignal - 输出汇总统计
        if isinstance(data, StopSignal):
            self._print_summary()
            if self._output_file:
                self._output_file.close()
            return data

        # 获取必要字段
        dataset: str = data.get("_dataset", "unknown")
        pred: str = data.get("generated", "")
        references: list[str] = data.get("references", [])
        all_classes: Optional[list[str]] = data.get("all_classes")
        length: int = data.get("length", 0)

        # 计算分数
        score = self._compute_score(pred, references, dataset, all_classes)

        # 分数 * 100（与原始 LongBench 一致）
        score_percent = round(score * 100, 2)

        # 收集分数
        self._scores.append(score)
        if dataset not in self._dataset_scores:
            self._dataset_scores[dataset] = []
        self._dataset_scores[dataset].append(score)

        # LongBench-E 分桶
        if self.longbench_e_buckets and length > 0:
            bucket = self._get_length_bucket(length)
            self._bucket_scores[bucket].append(score)

        # 收集时间数据（由 MapOperator 自动添加）
        if "refine_time" in data:
            self._refine_times.append(data["refine_time"])
        if "generate_time" in data:
            self._generate_times.append(data["generate_time"])
        if "retrieve_time" in data:
            self._retrieve_times.append(data["retrieve_time"])

        # 保存结果到 JSONL
        self._save_result(data, score, dataset)

        # 打印单个样本分数和时间
        metric_type = DATASET_TO_METRIC.get(dataset, "qa_f1")
        total_time = (
            data.get("retrieve_time", 0) + data.get("refine_time", 0) + data.get("generate_time", 0)
        )
        time_str = f" (time={total_time:.3f}s)" if total_time > 0 else ""
        print(f"\033[92m[LongBench {dataset}] {metric_type}: {score_percent}{time_str}\033[0m")

        # 将分数添加到数据中
        data["longbench_score"] = score
        data["longbench_score_percent"] = score_percent
        data["longbench_metric"] = metric_type

        return data

    def _print_summary(self) -> None:
        """打印汇总统计"""
        if not self._scores:
            print("\n" + "=" * 80)
            print("No LongBench samples processed")
            print("=" * 80)
            return

        print("\n" + "=" * 80)
        print(f"LONGBENCH EVALUATION SUMMARY ({len(self._scores)} samples)")
        print("=" * 80)

        # 总体平均分
        avg_score = sum(self._scores) / len(self._scores) * 100
        print(f"\033[92m[Overall Average Score]: {avg_score:.2f}\033[0m")

        # 按数据集分组的平均分
        if self._dataset_scores:
            print("\n--- Per-Dataset Scores ---")
            for dataset, scores in sorted(self._dataset_scores.items()):
                avg = sum(scores) / len(scores) * 100
                metric_type = DATASET_TO_METRIC.get(dataset, "qa_f1")
                print(f"  {dataset} ({metric_type}): {avg:.2f} ({len(scores)} samples)")

        # LongBench-E 分桶分数
        if self.longbench_e_buckets:
            print("\n--- LongBench-E Length Buckets ---")
            for bucket, scores in self._bucket_scores.items():
                if scores:
                    avg = sum(scores) / len(scores) * 100
                    print(f"  {bucket}: {avg:.2f} ({len(scores)} samples)")

        # 时间统计
        has_time_data = self._refine_times or self._generate_times or self._retrieve_times
        if has_time_data:
            print("\n--- Timing Statistics (seconds) ---")
            if self._retrieve_times:
                avg_retrieve = sum(self._retrieve_times) / len(self._retrieve_times)
                total_retrieve = sum(self._retrieve_times)
                print(
                    f"  Retrieve: avg={avg_retrieve:.3f}s, total={total_retrieve:.2f}s ({len(self._retrieve_times)} samples)"
                )
            if self._refine_times:
                avg_refine = sum(self._refine_times) / len(self._refine_times)
                total_refine = sum(self._refine_times)
                print(
                    f"  Refine:   avg={avg_refine:.3f}s, total={total_refine:.2f}s ({len(self._refine_times)} samples)"
                )
            if self._generate_times:
                avg_generate = sum(self._generate_times) / len(self._generate_times)
                total_generate = sum(self._generate_times)
                print(
                    f"  Generate: avg={avg_generate:.3f}s, total={total_generate:.2f}s ({len(self._generate_times)} samples)"
                )
            # 总时间
            total_time = (
                sum(self._retrieve_times) + sum(self._refine_times) + sum(self._generate_times)
            )
            print(f"  \033[92mTotal Pipeline Time: {total_time:.2f}s\033[0m")

        print("=" * 80 + "\n")

    def get_results(self) -> dict[str, Any]:
        """获取评估结果（用于程序化访问）"""
        results: dict[str, Any] = {
            "overall_score": (sum(self._scores) / len(self._scores) * 100 if self._scores else 0),
            "sample_count": len(self._scores),
            "per_dataset": {},
            "timing": {},
        }

        for dataset, scores in self._dataset_scores.items():
            results["per_dataset"][dataset] = {
                "score": sum(scores) / len(scores) * 100 if scores else 0,
                "count": len(scores),
                "metric": DATASET_TO_METRIC.get(dataset, "qa_f1"),
            }

        if self.longbench_e_buckets:
            results["buckets"] = {}
            for bucket, scores in self._bucket_scores.items():
                results["buckets"][bucket] = {
                    "score": sum(scores) / len(scores) * 100 if scores else 0,
                    "count": len(scores),
                }

        # 时间统计
        if self._retrieve_times:
            results["timing"]["retrieve"] = {
                "avg": sum(self._retrieve_times) / len(self._retrieve_times),
                "total": sum(self._retrieve_times),
                "count": len(self._retrieve_times),
            }
        if self._refine_times:
            results["timing"]["refine"] = {
                "avg": sum(self._refine_times) / len(self._refine_times),
                "total": sum(self._refine_times),
                "count": len(self._refine_times),
            }
        if self._generate_times:
            results["timing"]["generate"] = {
                "avg": sum(self._generate_times) / len(self._generate_times),
                "total": sum(self._generate_times),
                "count": len(self._generate_times),
            }

        return results

    def __del__(self) -> None:
        """对象销毁时关闭文件"""
        try:
            if self._output_file:
                self._output_file.close()
        except Exception:
            pass

"""
LongBench Batch Processing
==========================

LongBench 数据集批处理函数，从 THUDM/LongBench 加载数据并转换为 SAGE 标准格式。

迁移自 sage-libs/foundation/io/batch.py，遵循 SAGE 架构：
- benchmark 相关组件统一放在 sage-benchmark (L5)
"""

from typing import Any

from sage.libs.foundation.io import HFDatasetBatch


class LongBenchBatch(HFDatasetBatch):
    """
    LongBench 数据集批处理函数

    专门用于 LongBench 长文本理解基准测试，字段映射：
    - input → query（用户问题）
    - context → context（长文本上下文，LongBench 自带，无需检索）
    - answers → references（标准答案列表）
    - all_classes → all_classes（分类任务的类别列表）
    - length → length（原始文本长度，用于 LongBench-E 分桶评估）

    **与 SAGE RAG Pipeline 的对齐说明**：

    SAGE RAG 标准数据流:
    - query: 用户问题
    - references: 标准答案（评估用）
    - retrieval_results: 检索到的文档 List[Dict]（Retriever 输出）
    - refining_results: 压缩后的文档 List[str]（Refiner 输出）
    - context: 上下文字符串或列表（Promptor 读取）
    - generated: 生成的答案（Generator 输出）

    LongBench 特殊处理:
    - LongBench 自带 context，跳过 Retriever 阶段
    - context 直接作为 `context` 字段供 Promptor 使用
    - 同时设置 `retrieval_results` 为空列表（表示无检索）

    Input: None (直接从HF数据集读取)
    Output: SAGE RAGResponse 兼容格式 + LongBench 专用字段

    Config Keys (继承自 HFDatasetBatch):
        hf_dataset_name: str - 固定为 "THUDM/LongBench"
        hf_dataset_config: str - 如 "multi_news", "hotpotqa", "multi_news_e" 等
        hf_split: str - 默认 "test"
        max_samples: int - 最大样本数限制

    Output Fields (SAGE RAG 标准字段):
        query: str - 用户问题（来自 LongBench input）
        references: List[str] - 标准答案列表（来自 LongBench answers，评估用）
        context: str - 长文本上下文（来自 LongBench context，供 Promptor 使用）
        retrieval_results: List[Dict] - 空列表（LongBench 不走检索）

    Output Fields (LongBench 专用字段):
        all_classes: List[str] | None - 分类任务类别（trec, lsht 等）
        length: int - 原始文本长度（LongBench-E 分桶评估用）
        _dataset: str - 数据集名称（用于选择评估指标）
        _is_longbench_e: bool - 是否是 LongBench-E 版本
    """

    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__(config, **kwargs)
        # 解析数据集名称和是否为 LongBench-E
        self._dataset_name = self._parse_dataset_name()
        self._is_longbench_e = self._check_longbench_e()

    def _parse_dataset_name(self) -> str:
        """从 hf_dataset_config 解析数据集名称（去除 _e 后缀）"""
        config_name = self.hf_config or ""
        if config_name.endswith("_e"):
            return config_name[:-2]  # 去除 _e 后缀
        return config_name

    def _check_longbench_e(self) -> bool:
        """检查是否是 LongBench-E 版本"""
        config_name = self.hf_config or ""
        return config_name.endswith("_e")

    def _build_iter(self):
        """构建 LongBench 数据集迭代器，重写父类方法"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for LongBenchBatch. "
                "Install with: pip install datasets"
            )

        ds = load_dataset(self.hf_name, self.hf_config, split=self.hf_split, streaming=True)
        for ex in ds:
            if isinstance(ex, dict):
                yield self._transform_example(ex)

    def _transform_example(self, ex: dict[str, Any]) -> dict[str, Any]:
        """
        LongBench 字段映射到 SAGE RAG Pipeline 标准格式

        LongBench 原始字段 → SAGE RAG 标准字段:
        - input → query（用户问题）
        - context → context（上下文，供 Promptor 使用）
        - answers → references（标准答案，供 Evaluate 使用）

        LongBench 专用字段保留:
        - all_classes（分类任务类别）
        - length（原始长度，LongBench-E 分桶用）
        """
        return {
            # ========== SAGE RAG Pipeline 标准字段 ==========
            "query": ex.get("input", ""),
            "references": ex.get("answers") or [],
            "context": ex.get("context", ""),
            # 空列表表示跳过检索阶段（LongBench 自带 context）
            "retrieval_results": [],
            # ========== LongBench 专用字段 ==========
            "all_classes": ex.get("all_classes"),
            "length": ex.get("length", 0),
            # ========== 内部元数据（下划线前缀，pipeline 流转用）==========
            "_dataset": self._dataset_name,
            "_is_longbench_e": self._is_longbench_e,
        }

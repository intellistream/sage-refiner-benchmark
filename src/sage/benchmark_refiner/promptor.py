"""
LongBench Promptor - 优化版

改进点：
1. 从 JSON 文件加载配置（而非硬编码）
2. 支持模型专用 chat template（照搬官方 build_chat）
3. 中间截断策略（照搬官方实现）
4. 数据集验证
"""

import json
from pathlib import Path
from typing import Any, Optional

from sage.common.core.functions import MapFunction as MapOperator

from .constants import NO_CHAT_DATASETS, SUPPORTED_DATASETS
from .utils import build_chat, truncate_middle


class LongBenchPromptor(MapOperator):
    """
    LongBench 专用 Promptor。

    功能：
    1. 从 JSON 文件加载任务专用 prompt 模板
    2. 使用 context 和 input (query) 填充模板
    3. Token 级中间截断（当超过 max_input_tokens 时，保留首尾）
    4. 按数据集决定是否使用 chat template（few-shot 和代码任务不使用）

    配置参数：
        - max_input_tokens: int | None - 最大输入 token 数，超过则中间截断
        - is_chat_model: bool - 是否使用 chat template（默认 False）
        - model_name_or_path: str | None - 模型路径，用于加载 tokenizer
    """

    # 配置文件缓存（类级别，避免重复加载）
    _prompt_templates: Optional[dict[str, str]] = None
    _max_gen_lengths: Optional[dict[str, int]] = None
    _model_max_lengths: Optional[dict[str, int]] = None

    @classmethod
    def _load_configs(cls) -> None:
        """延迟加载 JSON 配置文件"""
        if cls._prompt_templates is None:
            config_dir = Path(__file__).parent / "config"
            with open(config_dir / "dataset2prompt.json", encoding="utf-8") as f:
                cls._prompt_templates = json.load(f)
            with open(config_dir / "dataset2maxlen.json", encoding="utf-8") as f:
                cls._max_gen_lengths = json.load(f)
            with open(config_dir / "model2maxlen.json", encoding="utf-8") as f:
                cls._model_max_lengths = json.load(f)

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_configs()

        self.config = config
        self.max_input_tokens: Optional[int] = config.get("max_input_tokens")
        self.is_chat_model: bool = config.get("is_chat_model", False)
        self.model_name: str = config.get("model_name_or_path", "")

        # 延迟加载 tokenizer
        self._tokenizer: Optional[Any] = None

    @property
    def tokenizer(self) -> Optional[Any]:
        """延迟加载 tokenizer"""
        if self._tokenizer is None and self.model_name:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self.logger.info(f"Loaded tokenizer from {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load tokenizer: {e}")
        return self._tokenizer

    def execute(self, data: dict[str, Any]) -> list[Any]:
        """
        生成 LongBench 风格的 prompt。

        输入格式（来自 LongBenchBatch 或经过 Refiner）：
        {
            "query": str,           # 用户问题（原 input 字段）
            "context": str,         # 长文本上下文（原始）
            "refining_results": List[str],  # 压缩后的上下文（可选，Refiner 输出）
            "references": list,     # 标准答案
            "_dataset": str,        # 数据集名称
            ...
        }

        **上下文选择优先级**：
        1. refining_results（如果存在且非空，来自 Refiner 压缩后的结果）
        2. context（原始上下文，LongBench 自带）

        输出格式：
        [original_data, prompt_string]

        Args:
            data: 包含 query, context, _dataset 等字段的字典

        Returns:
            [原始数据, prompt 字符串] 列表
        """
        dataset = data.get("_dataset", "")
        query = data.get("query", "")

        # 数据集验证
        if dataset and dataset not in SUPPORTED_DATASETS:
            self.logger.warning(
                f"Unknown dataset '{dataset}', using default template. "
                f"Supported: {SUPPORTED_DATASETS}"
            )

        # 上下文选择优先级：refining_results > context
        refining_results = data.get("refining_results", [])
        if refining_results:
            if isinstance(refining_results, list):
                context = "\n\n".join(refining_results)
            else:
                context = str(refining_results)
            self.logger.info("Using refining_results (compressed context)")
        else:
            context = data.get("context", "")

        # 1. 获取数据集专用模板（从 JSON 加载）
        assert self._prompt_templates is not None
        template = self._prompt_templates.get(dataset, "{context}\n\nQuestion: {input}\nAnswer:")

        # 2. 填充模板（LongBench 使用 {context} 和 {input} 占位符）
        prompt = template.format(context=context, input=query)

        # 3. 中间截断（如果配置了 max_input_tokens）
        if self.max_input_tokens and self.tokenizer:
            prompt = truncate_middle(prompt, self.tokenizer, self.max_input_tokens)

        # 4. Chat template（按数据集决定）
        # 原始 pred.py: if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        #                   prompt = build_chat(tokenizer, prompt, model_name)
        if self.is_chat_model and dataset not in NO_CHAT_DATASETS:
            prompt = build_chat(prompt, self.model_name, self.tokenizer)

        # 5. 设置 max_gen_tokens 供 Generator 使用
        assert self._max_gen_lengths is not None
        data["_max_gen_tokens"] = self._max_gen_lengths.get(dataset, 128)

        self.logger.info(f"dataset={dataset}, prompt_length={len(prompt)}")
        return [data, prompt]

    @classmethod
    def get_max_gen_length(cls, dataset: str) -> int:
        """
        获取数据集的最大生成长度。

        Args:
            dataset: 数据集名称

        Returns:
            最大生成 token 数
        """
        cls._load_configs()
        assert cls._max_gen_lengths is not None
        return cls._max_gen_lengths.get(dataset, 128)

    @classmethod
    def get_prompt_template(cls, dataset: str) -> str:
        """
        获取数据集的 prompt 模板。

        Args:
            dataset: 数据集名称

        Returns:
            prompt 模板字符串
        """
        cls._load_configs()
        assert cls._prompt_templates is not None
        return cls._prompt_templates.get(dataset, "{context}\n\nQuestion: {input}\nAnswer:")

    @classmethod
    def get_model_max_length(cls, model_name: str, default: int = 8192) -> int:
        """
        获取模型的最大上下文长度。

        支持精确匹配和模糊匹配（模型名称包含关系）。

        Args:
            model_name: 模型名称或路径
            default: 默认最大长度（如果模型未在映射中）

        Returns:
            模型最大上下文 token 数
        """
        cls._load_configs()
        assert cls._model_max_lengths is not None

        # 精确匹配
        if model_name in cls._model_max_lengths:
            return cls._model_max_lengths[model_name]

        # 模糊匹配
        model_lower = model_name.lower()
        for known_model, max_len in cls._model_max_lengths.items():
            if known_model.lower() in model_lower:
                return max_len

        return default

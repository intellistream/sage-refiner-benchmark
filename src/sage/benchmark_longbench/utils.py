"""
LongBench 工具函数

包含模型专用的 chat template 和后处理逻辑（照搬官方 pred.py）
"""

from typing import Any, Optional


def build_chat(prompt: str, model_name: str, tokenizer: Optional[Any] = None) -> str:
    """
    构建模型专用的 chat prompt。

    照搬自官方 pred.py 的 build_chat 函数。

    Args:
        prompt: 原始 prompt
        model_name: 模型名称
        tokenizer: tokenizer 实例（用于 chatglm3）

    Returns:
        包装后的 prompt
    """
    model_name_lower = model_name.lower()

    if "chatglm3" in model_name_lower and tokenizer:
        # chatglm3 使用 tokenizer 的专用方法
        return tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name_lower and tokenizer:
        return tokenizer.build_prompt(prompt)
    elif "llama2" in model_name_lower:
        return f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name_lower:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        return header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name_lower:
        return f"<|User|>:{prompt}<eoh>\n<|Bot|>:"

    # 默认：尝试使用 tokenizer 的 apply_chat_template
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    return prompt


def post_process(response: str, model_name: str) -> str:
    """
    模型专用输出后处理。

    照搬自官方 pred.py 的 post_process 函数。

    Args:
        response: 模型原始输出
        model_name: 模型名称

    Returns:
        处理后的输出
    """
    model_name_lower = model_name.lower()

    if "xgen" in model_name_lower:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name_lower:
        response = response.split("<eoa>")[0]

    return response


def truncate_middle(
    prompt: str,
    tokenizer: Any,
    max_length: int,
) -> str:
    """
    中间截断策略（保留首尾）。

    照搬自官方 pred.py：
    # truncate to fit max_length (we suggest truncate in the middle,
    # since the left and right side may contain crucial instructions)

    Args:
        prompt: 原始 prompt
        tokenizer: tokenizer 实例
        max_length: 最大 token 数

    Returns:
        截断后的 prompt
    """
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

    if len(tokenized) > max_length:
        half = max_length // 2
        prompt = tokenizer.decode(tokenized[:half], skip_special_tokens=True) + tokenizer.decode(
            tokenized[-half:], skip_special_tokens=True
        )

    return prompt

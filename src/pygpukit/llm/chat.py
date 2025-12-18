"""
Chat Template Support for PyGPUkit LLM

Provides chat message formatting for instruction-following models.
Works with HuggingFace tokenizers for model-specific templates.

Usage:
    from pygpukit.llm.chat import ChatMessage, apply_chat_template

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!"),
    ]

    # With HuggingFace tokenizer (recommended)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    input_ids = apply_chat_template(messages, tokenizer)

    # Or get formatted text
    text = format_chat_messages(messages, model_type="qwen3")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from typing import TypeAlias

    Messages: TypeAlias = Union[list["ChatMessage"], list[dict[str, str]]]


@dataclass
class ChatMessage:
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message sender ("system", "user", or "assistant")
        content: The text content of the message
    """

    role: str  # "system", "user", "assistant"
    content: str


def _normalize_messages(messages: Messages) -> list[dict[str, str]]:
    """Convert messages to list of dicts format."""
    result = []
    for msg in messages:
        if isinstance(msg, ChatMessage):
            result.append({"role": msg.role, "content": msg.content})
        else:
            result.append(msg)
    return result


# =============================================================================
# Model-specific Templates
# =============================================================================

# Qwen3 / Qwen2 Chat template
# fmt: off
QWEN_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""  # noqa: E501

# LLaMA 2 Chat template
LLAMA2_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {% if loop.first and system_message %}<<SYS>>
{{ system_message }}
<</SYS>>

{% endif %}{{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %} {{ message['content'] }}</s>{% endif %}{% endfor %}"""  # noqa: E501

# LLaMA 3 Chat template
LLAMA3_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|}

{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}"""  # noqa: E501

# Mistral Instruct template
MISTRAL_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}{% endfor %}"""  # noqa: E501
# fmt: on

# ChatML template (generic, used by many models)
CHATML_TEMPLATE = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# Template mapping
TEMPLATES = {
    "qwen": QWEN_TEMPLATE,
    "qwen2": QWEN_TEMPLATE,
    "qwen3": QWEN_TEMPLATE,
    "llama2": LLAMA2_TEMPLATE,
    "llama3": LLAMA3_TEMPLATE,
    "mistral": MISTRAL_TEMPLATE,
    "chatml": CHATML_TEMPLATE,
}


def format_chat_messages(
    messages: Messages,
    model_type: str = "chatml",
    add_generation_prompt: bool = True,
) -> str:
    """Format chat messages using a model-specific template.

    This function uses Jinja2 templates to format messages according to
    the model's expected chat format.

    Args:
        messages: List of ChatMessage objects or dicts with 'role' and 'content'
        model_type: Model type for template selection ("qwen", "llama2", "llama3",
                   "mistral", "chatml")
        add_generation_prompt: Whether to add the assistant prompt at the end

    Returns:
        Formatted string ready for tokenization

    Example:
        >>> messages = [
        ...     ChatMessage(role="user", content="Hello!")
        ... ]
        >>> text = format_chat_messages(messages, model_type="qwen3")
        >>> print(text)
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
    """
    try:
        from jinja2 import Template
    except ImportError as e:
        raise ImportError(
            "jinja2 is required for chat template formatting. Install it with: pip install jinja2"
        ) from e

    template_str = TEMPLATES.get(model_type.lower(), CHATML_TEMPLATE)
    template = Template(template_str)

    msgs = _normalize_messages(messages)
    return template.render(messages=msgs, add_generation_prompt=add_generation_prompt)


def apply_chat_template(
    messages: Messages,
    tokenizer: Any,
    add_generation_prompt: bool = True,
    return_tensors: str | None = None,
) -> list[int]:
    """Apply chat template and tokenize using HuggingFace tokenizer.

    This is the recommended way to format chat messages when using a
    HuggingFace tokenizer, as it uses the tokenizer's built-in chat_template.

    Args:
        messages: List of ChatMessage objects or dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer with apply_chat_template method
        add_generation_prompt: Whether to add the assistant prompt at the end
        return_tensors: Not used (kept for API compatibility)

    Returns:
        List of token IDs

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"},
        ... ]
        >>> input_ids = apply_chat_template(messages, tokenizer)
    """
    msgs = _normalize_messages(messages)

    # Try HuggingFace tokenizer's apply_chat_template first
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )

    # Fallback: detect model type and use our templates
    model_type = "chatml"  # default

    # Try to detect model type from tokenizer
    if hasattr(tokenizer, "name_or_path"):
        name = tokenizer.name_or_path.lower()
        if "qwen" in name:
            model_type = "qwen"
        elif "llama-3" in name or "llama3" in name:
            model_type = "llama3"
        elif "llama-2" in name or "llama2" in name:
            model_type = "llama2"
        elif "mistral" in name:
            model_type = "mistral"

    formatted = format_chat_messages(msgs, model_type, add_generation_prompt)
    return tokenizer.encode(formatted, add_special_tokens=False)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_chat_prompt(
    user_message: str,
    system_message: str | None = None,
    assistant_prefix: str | None = None,
) -> list[ChatMessage]:
    """Create a simple chat prompt with optional system message.

    Args:
        user_message: The user's message
        system_message: Optional system prompt
        assistant_prefix: Optional prefix for assistant response (for constrained generation)

    Returns:
        List of ChatMessage objects

    Example:
        >>> messages = create_chat_prompt(
        ...     "What is 2+2?",
        ...     system_message="You are a math tutor."
        ... )
    """
    messages = []
    if system_message:
        messages.append(ChatMessage(role="system", content=system_message))
    messages.append(ChatMessage(role="user", content=user_message))
    if assistant_prefix:
        messages.append(ChatMessage(role="assistant", content=assistant_prefix))
    return messages

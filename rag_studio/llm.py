"""LLM abstraction layer supporting local Ollama models."""

import logging
from typing import Optional

import ollama
from rag_studio.config import Config


def llm_chat(messages: list[dict], model: Optional[str] = None) -> str:
    """Sends a chat sequence to the configured LLM and returns the text response.

    Args:
        messages (list[dict]): A list of message dictionaries (e.g., {"role": "...", "content": "..."}).
        model (Optional[str]): The model to use. Defaults to Config.LLM_MODEL.

    Returns:
        str: The generated text response from the model.

    Raises:
        ValueError: If the configured provider is not 'ollama'.
    """
    model = model or Config.LLM_MODEL
    provider = Config.LLM_PROVIDER

    logger = logging.getLogger("rag_studio")
    logger.info(f"Calling {provider} ({model})...")

    if provider != "ollama":
        raise ValueError(f"Unsupported or non-free LLM provider: {provider}")

    response = ollama.chat(model=model, messages=messages)
    content = response["message"]["content"].strip()
    
    logger.info(f"{provider} responded ({len(content)} chars)")
    return content

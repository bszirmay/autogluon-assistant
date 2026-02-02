import logging
import os
from typing import Dict, List

from langchain_openai import ChatOpenAI

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantPortKeyChatOpenAI(ChatOpenAI, BaseAssistantChat):
    """Portkey-proxied OpenAI chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_name, "provider": "openai_portkey"}


def create_portkey_openai_chat(config, session_name: str) -> AssistantPortKeyChatOpenAI:
    """Create a Portkey-backed OpenAI chat model instance.

    Required environment variables:
    - PORTKEY_API_KEY: Portkey API key
    - PORTKEY_OPENAI_VIRTUAL_KEY: Portkey virtual key for OpenAI-compatible routing

    Optional:
    - PORTKEY_BASE_URL: Base URL for Portkey gateway (overridden by config.proxy_url if provided)

    See: https://portkey.ai/docs/introduction/what-is-portkey
    """
    model = config.model

    if "PORTKEY_API_KEY" not in os.environ:
        raise ValueError("PORTKEY_API_KEY not found in environment")
    if "PORTKEY_OPENAI_VIRTUAL_KEY" not in os.environ:
        raise ValueError("PORTKEY_OPENAI_VIRTUAL_KEY not found in environment")

    base_url = getattr(config, "proxy_url", None) or os.environ.get("PORTKEY_BASE_URL")
    if not base_url:
        raise ValueError("Portkey base URL not provided. Set llm.proxy_url or PORTKEY_BASE_URL.")

    logger.info(f"Using Portkey OpenAI-compatible model: {model} via {base_url} for session: {session_name}")

    kwargs = {
        "model_name": model,
        "api_key": os.environ["PORTKEY_OPENAI_VIRTUAL_KEY"],
        "base_url": base_url,
        "default_headers": {
            "x-portkey-api-key": os.getenv("PORTKEY_API_KEY"),
            "x-portkey-provider": "openai",
            "x-portkey-virtual-key": os.getenv("PORTKEY_OPENAI_VIRTUAL_KEY"),
        },
        "session_name": session_name,
        "max_tokens": config.max_tokens,
    }

    return AssistantPortKeyChatOpenAI(**kwargs)

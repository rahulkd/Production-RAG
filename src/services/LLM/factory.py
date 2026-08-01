from functools import lru_cache

from src.config import get_settings
from src.services.LLM.client import BedrockClient

## ollama
#from src.services.LLM.client_ollama import OllamaClient


@lru_cache(maxsize=1)
def make_bedrock_client() -> BedrockClient:
    """
    Create and return a singleton Bedrock client instance.

    Returns:
        BedrockClient: Configured AWS Bedrock client
    """
    settings = get_settings()
    return BedrockClient(settings)


## ollama
#@lru_cache(maxsize=1)
#def make_ollama_client() -> OllamaClient:
#    """
#    Create and return a singleton Ollama client instance.
#
#    Returns:
#        OllamaClient: Configured Ollama client
#    """
#    settings = get_settings()
#    return OllamaClient(settings)

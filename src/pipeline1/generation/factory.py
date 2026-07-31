import os

from src.pipeline1.generation.ollama_generator import OllamaGenerator


def build_generator(config):
    provider = getattr(config, "provider", "ollama")
    if provider == "mistral":
        from src.pipeline1.generation.mistral_generator import MistralGenerator
        return MistralGenerator(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
        )
    if provider == "openai":
        from src.pipeline1.generation.openai_generator import OpenAIGenerator
        return OpenAIGenerator(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            reasoning_effort=config.reasoning_effort,
            timeout_s=config.timeout_s,
        )
    return OllamaGenerator(
        model_name=config.model_name,
        base_url=os.getenv("OLLAMA_BASE_URL", config.base_url),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )

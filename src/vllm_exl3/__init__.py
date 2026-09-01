"""Register the routed-expert EXL3 implementation with a local vLLM runtime."""


def register() -> None:
    # Importing the module executes its register_quantization_config decorator.
    from . import exl3 as _exl3  # noqa: F401

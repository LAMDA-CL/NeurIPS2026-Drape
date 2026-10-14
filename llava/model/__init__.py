from .language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
# from .language_model.llava_mpt import LlavaMPTConfig, LlavaMPTForCausalLM
try:
    from .language_model.llava_mpt import LlavaMPTConfig, LlavaMPTForCausalLM
except ImportError:
    # Keep LLaMA path usable when MPT-specific transformers internals are unavailable.
    LlavaMPTConfig = None
    LlavaMPTForCausalLM = None


from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# MODEL = "TheBloke/llama2_70b_chat_uncensored-GPTQ"

MODEL = "TheBloke/Llama-2-7b-Chat-GPTQ"

def download_model() -> tuple:
    """Download the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(MODEL,
            use_safetensors=True,
            trust_remote_code=False,
            device="cuda:0",
            use_triton=False,
            quantize_config=None,
            inject_fused_attention=False)
    return model, tokenizer

if __name__ == "__main__":
    download_model()
    

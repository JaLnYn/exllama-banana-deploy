from potassium import Potassium, Request, Response

from transformers import pipeline
import torch
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
import os, glob


app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline('fill-mask', model='bert-base-uncased', device=device)

    model_directory =  "/mnt/str/models/llama-13b-4bit-128g/"

    # Locate files we need within that directory

    tokenizer_path = os.path.join(model_directory, "tokenizer.model")
    model_config_path = os.path.join(model_directory, "config.json")
    st_pattern = os.path.join(model_directory, "*.safetensors")
    model_path = glob.glob(st_pattern)

    # Create config, model, tokenizer and generator

    config = ExLlamaConfig(model_config_path)               # create config from config.json
    config.model_path = model_path                          # supply path to model weights file

    model = ExLlama(config)                                 # create ExLlama instance and load the weights
    tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

    cache = ExLlamaCache(model)                             # create cache for inference
    generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

# Configure generator

    generator.disallow_tokens([tokenizer.eos_token_id])

    generator.settings.token_repetition_penalty_max = 1.2
    generator.settings.temperature = 0.95
    generator.settings.top_p = 0.65
    generator.settings.top_k = 100
    generator.settings.typical = 0.5
   
    context = {
        "model": generator
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    generator = context.get("model")
    outputs = model(prompt)

    output = generator.generate_simple(prompt, max_new_tokens = 200)

    return Response(
        json = {"outputs": output[len(prompt):]}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()

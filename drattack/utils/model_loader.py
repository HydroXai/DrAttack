from copy import deepcopy

import torch
import torch.multiprocessing as mp
from fastchat.model import get_conversation_template
from transformers import (AutoModelForCausalLM, AutoTokenizer)

from .GPTWrapper import GPTAPIWrapper
from .GeminiWrapper import GeminiAPIWrapper

class ModelWorker(object):

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template, device):
        
        if "vicuna" in model_path:
            self.model_name = "vicuna"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                **model_kwargs
            )
            self.tokenizer = tokenizer
            self.conv_template = conv_template
        elif "llama" in model_path:
            self.model_name = "llama"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                **model_kwargs
            )
            self.tokenizer = tokenizer
            self.conv_template = conv_template
        elif "gpt" in model_path:
            self.model_name = "gpt"
            self.model = GPTAPIWrapper(model=model_path)  # gpt-3.5-turbo
            self.tokenizer = lambda x: x
            self.conv_template = "You are a helpful assistant."
        elif "gemini" in model_path:
            self.model_name = "gemini"
            self.model = GeminiAPIWrapper(model_name=model_path)  # gemini
            self.tokenizer = lambda x: x
            self.conv_template = "You are a helpful assistant."

    def run_task(self, ob, fn, *args, **kwargs):
        if fn == "grad":
            with torch.enable_grad():
                return ob.grad(*args, **kwargs)
        else:
            with torch.no_grad():
                if fn == "logits":
                    return ob.logits(*args, **kwargs)
                elif fn == "contrast_logits":
                    return ob.contrast_logits(*args, **kwargs)
                elif fn == "test":
                    return ob.test(*args, **kwargs)
                elif fn == "test_loss":
                    return ob.test_loss(*args, **kwargs)
                else:
                    return fn(*args, **kwargs)

    def stop(self):
        torch.cuda.empty_cache()

    def __call__(self, ob, fn, *args, **kwargs):
        return self.run_task(deepcopy(ob), fn, *args, **kwargs)

def get_worker(params, eval=False):

    if ('gpt' not in params.model_path) and ('gemini' not in params.model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            params.tokenizer_path,
            trust_remote_code=True,
            **params.tokenizer_kwarg
        )
        if 'oasst-sft-6-llama-30b' in params.tokenizer_path:
            tokenizer.bos_token_id = 1
            tokenizer.unk_token_id = 0
        if 'guanaco' in params.tokenizer_path:
            tokenizer.eos_token_id = 2
            tokenizer.unk_token_id = 0
        if 'llama-2' in params.tokenizer_path:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'falcon' in params.tokenizer_path:
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loaded tokenizer")

        raw_conv_template = get_conversation_template(params.conversation_template)

        if raw_conv_template.name == 'zero_shot':
            raw_conv_template.roles = tuple(['### ' + r for r in raw_conv_template.roles])
            raw_conv_template.sep = '\n'
        elif raw_conv_template.name == 'llama-2':
            raw_conv_template.sep2 = raw_conv_template.sep2.strip()

        conv_template = raw_conv_template
        print(f"Loaded conversation template")
        worker = ModelWorker(
            params.model_path,
            params.model_kwarg,
            tokenizer,
            conv_template,
            params.device
        )

        print('Loaded target LLM model')
    
    else:
        tokenizer = [None]
        worker = ModelWorker(
            params.model_path,
            None,
            None,
            None,
            None
        )
    return worker

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4-0125-preview":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-4-1106-preview":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-4-turbo":{
            "path":"gpt-4-turbo",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "gpt-3.5-turbo-1106": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        },
        "claude-3-sonnet-20240229":{
            "path":"claude-3-sonnet-20240229",
            "template":"claude-3-sonnet-20240229"
        },
        "claude-3-haiku-20240307":{
            "path":"claude-3-haiku-20240307",
            "template":"claude-2"
        },
        "claude-3-opus-20240229":{
            "path":"claude-3-opus-20240229",
            "template":"claude-3-opus-20240229"
        },
        "gemini-pro": {
            "path": "gemini-pro",
            "template": "gemini-pro"
        },
        "/media/d1/huggingface.co/models/meta-llama/Llama-2-7b-chat-hf":{
            "path":"/media/d1/huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
            "template":"llama-2"
        },
        "/media/d1/huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2":{
            "path":"/media/d1/huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            "template":"mistral"
        },
        "/media/d1/huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct":{
            "path":"/media/d1/huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
            "template":"llama-2"
        },
        "/media/d1/huggingface.co/models/HuggingFaceM4/tiny-random-LlamaForCausalLM":{
            "path":"media/d1/huggingface.co/models/HuggingFaceM4/tiny-random-LlamaForCausalLM",
            "template":"llama-2"
        },
        "/media/d1/huggingface.co/models/meta-llama/LlamaGuard-7b":{
            "path":"/media/d1/huggingface.co/models/meta-llama/LlamaGuard-7b",
            "template":"llama-2"
        }
    }
    assert model_name in full_model_dict, f"Model {model_name} not found in `full_model_dict` (available keys {full_model_dict.keys()})"
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template    

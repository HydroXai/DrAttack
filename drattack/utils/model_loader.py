import torch

from copy import deepcopy


class ModelWorker(object):
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, ob, fn, *args, **kwargs):
        return self.run_task(deepcopy(ob), fn, *args, **kwargs)

    def get_embeddings(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        return input_ids

    def infer(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant." },
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(input_ids, do_sample=True, pad_token_id=self.tokenizer.pad_token_id, temperature=0.8, max_length=1024)
        outputs = generated_ids[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return response

    def run_task(self, ob, fn, *args, **kwargs):
        if fn == "grad":
            with torch.enable_grad():
                return ob.grad(*args, **kwargs)
        with torch.no_grad():
            if fn == "logits":
                return ob.logits(*args, **kwargs)
            elif fn == "contrast_logits":
                return ob.contrast_logits(*args, **kwargs)
            elif fn == "test":
                return ob.test(*args, **kwargs)
            elif fn == "test_loss":
                return ob.test_loss(*args, **kwargs)
            return fn(*args, **kwargs)

    def stop(self):
        torch.cuda.empty_cache()

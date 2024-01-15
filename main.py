from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


print('loading tokenizer')

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)

# These are the params passed in by oobabooga. If I just pass `trust_remote_code` by itself we run out of memory. Need to dig into what these all do and which one fixed the issue.
# Also had to pip install accellerate per error "ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate"
params = {
    'low_cpu_mem_usage': True,
    'trust_remote_code': False, 
    'torch_dtype': torch.float16, 
    'use_safetensors': None,
    'device_map': 'auto', 
    'max_memory': None,
    'rope_scaling': {
        'type': 'linear',
        'factor': 4
    }
}

print('loading model')
#model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True).cuda()
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", **params).cuda()

print('loaded')
messages=[
    { 'role': 'user', 'content': "write a quick sort algorithm in python."}
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
print('generating')
# 32021 is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=32021)

print('decoding')
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))

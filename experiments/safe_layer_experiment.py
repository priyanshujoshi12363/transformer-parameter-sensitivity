import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.expanduser("~/cpu-os/ai_experiment/models/qwen2.5-3b")
TARGET_LAYER = 17
SCALE = 0.90
MAX_NEW_TOKENS = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()

layer = model.model.layers[TARGET_LAYER]

print(f"Target layer: {TARGET_LAYER}")

for name, parameter in layer.named_parameters():
    print(
        name,
        "shape=",
        tuple(parameter.shape),
        "device=",
        parameter.device,
        "dtype=",
        parameter.dtype
    )

backups = {}

for name, parameter in layer.named_parameters():
    backups[name] = parameter.detach().clone()

print("Backup created.")

def checksum(tensor):
    return tensor.float().sum().item()

original_checksums = {}

for name, parameter in layer.named_parameters():
    original_checksums[name] = checksum(parameter)

print("Original checksums saved.")

with torch.no_grad():
    for name, parameter in layer.named_parameters():
        parameter.mul_(SCALE)

print(f"Layer {TARGET_LAYER} scaled by {SCALE}")

question = input("\nEnter your question: ")

messages = [
    {
        "role": "user",
        "content": question
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

if hasattr(inputs, "input_ids"):
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
else:
    input_ids = inputs
    attention_mask = None

input_device = next(
    parameter.device
    for parameter in model.parameters()
    if parameter.device.type != "meta"
)

input_ids = input_ids.to(input_device)

if attention_mask is not None:
    attention_mask = attention_mask.to(input_device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

new_tokens = outputs[0][input_ids.shape[-1]:]

answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("\nAnswer:\n")
print(answer)

with torch.no_grad():
    for name, parameter in layer.named_parameters():
        parameter.copy_(backups[name])

print("\nLayer restored.")

restoration_ok = True

for name, parameter in layer.named_parameters():
    restored_checksum = checksum(parameter)

    if not torch.equal(parameter, backups[name]):
        restoration_ok = False
        print(f"RESTORE FAILED: {name}")

    if restored_checksum != original_checksums[name]:
        restoration_ok = False
        print(f"CHECKSUM MISMATCH: {name}")

if restoration_ok:
    print("Restoration verified successfully.")
else:
    print("WARNING: Restoration verification failed.")

print("Original Safetensors files were never modified.")
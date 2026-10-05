import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
from threading import Thread

MODEL_PATH = "models/qwen2.5-3b"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("Model loaded!")
print("Device map:")
print(model.hf_device_map)

while True:
    question = input("\nYou: ")

    if question.lower() in {"exit", "quit"}:
        break

    messages = [
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )

    thread.start()

    print("\nQwen: ", end="", flush=True)

    for token in streamer:
        print(token, end="", flush=True)

    print()

    thread.join()
import os
import json
import math
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.expanduser(
    "~/cpu-os/ai_experiment/models/qwen2.5-3b"
)

BENCHMARK_PATH = os.path.join(
    os.path.dirname(__file__),
    "benchmark.json"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

SCALE = 0.90
MAX_NEW_TOKENS = 256
LAYERS = list(range(36))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading benchmark...")

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

questions = []

for category, items in benchmark.items():
    for question in items:
        questions.append({
            "category": category,
            "question": question
        })

print(f"Loaded {len(questions)} benchmark questions.")

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()

print("Model loaded.")

input_device = next(
    parameter.device
    for parameter in model.parameters()
    if parameter.device.type != "meta"
)

print(f"Input device: {input_device}")


def prepare_prompt(question):
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    if hasattr(result, "input_ids"):
        input_ids = result["input_ids"]
        attention_mask = result.get("attention_mask", None)
    else:
        input_ids = result
        attention_mask = None

    input_ids = input_ids.to(input_device)

    if attention_mask is not None:
        attention_mask = attention_mask.to(input_device)

    return input_ids, attention_mask


def generate_answer(question):
    input_ids, attention_mask = prepare_prompt(question)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    generated_tokens = output[0][input_ids.shape[-1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer


def calculate_answer_loss(question, answer):
    prompt_ids, _ = prepare_prompt(question)

    answer_tokens = tokenizer(
        answer,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].to(input_device)

    combined = torch.cat(
        [prompt_ids, answer_tokens],
        dim=1
    )

    labels = combined.clone()

    labels[:, :prompt_ids.shape[1]] = -100

    attention_mask = torch.ones_like(combined)

    with torch.no_grad():
        outputs = model(
            input_ids=combined,
            attention_mask=attention_mask,
            labels=labels
        )

    loss = outputs.loss.item()

    return loss


def backup_layer(layer):
    return {
        name: parameter.detach().clone()
        for name, parameter in layer.named_parameters()
    }


def restore_layer(layer, backup):
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            parameter.copy_(backup[name])


def perturb_layer(layer, scale):
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.mul_(scale)


print("\n" + "=" * 70)
print("BASELINE EXPERIMENT")
print("=" * 70)

baseline_results = []

baseline_start = time.time()

for index, item in enumerate(questions, 1):

    print(
        f"\nBaseline {index}/{len(questions)} "
        f"[{item['category']}]"
    )

    answer = generate_answer(
        item["question"]
    )

    loss = calculate_answer_loss(
        item["question"],
        answer
    )

    baseline_results.append({
        "id": index,
        "category": item["category"],
        "question": item["question"],
        "answer": answer,
        "baseline_loss": loss
    })

    print(f"Baseline loss: {loss:.6f}")
    print(f"Answer: {answer[:200]}")

baseline_time = time.time() - baseline_start

baseline_file = os.path.join(
    RESULTS_DIR,
    "baseline.json"
)

with open(
    baseline_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        baseline_results,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"\nBaseline saved to: {baseline_file}"
)

all_results = []

print("\n" + "=" * 70)
print("LAYER SENSITIVITY EXPERIMENT")
print("=" * 70)

for layer_index in LAYERS:

    print("\n" + "-" * 70)
    print(
        f"Testing Layer {layer_index} "
        f"with scale={SCALE}"
    )
    print("-" * 70)

    layer = model.model.layers[layer_index]

    backup = backup_layer(layer)

    layer_results = []

    try:

        perturb_layer(
            layer,
            SCALE
        )

        for question_index, baseline in enumerate(
            baseline_results,
            1
        ):

            print(
                f"Layer {layer_index}: "
                f"{question_index}/{len(baseline_results)} "
                f"[{baseline['category']}]"
            )

            perturbed_loss = calculate_answer_loss(
                baseline["question"],
                baseline["answer"]
            )

            delta_nll = (
                perturbed_loss
                - baseline["baseline_loss"]
            )

            perturbed_answer = generate_answer(
                baseline["question"]
            )

            layer_results.append({
                "id": baseline["id"],
                "category": baseline["category"],
                "question": baseline["question"],
                "baseline_loss": baseline["baseline_loss"],
                "perturbed_loss": perturbed_loss,
                "delta_nll": delta_nll,
                "baseline_answer": baseline["answer"],
                "perturbed_answer": perturbed_answer
            })

            print(
                f"Baseline loss: {baseline['baseline_loss']:.6f}"
            )

            print(
                f"Perturbed loss: {perturbed_loss:.6f}"
            )

            print(
                f"Delta NLL: {delta_nll:.6f}"
            )

    finally:

        restore_layer(
            layer,
            backup
        )

        del backup

        print(
            f"Layer {layer_index} restored."
        )

    layer_file = os.path.join(
        RESULTS_DIR,
        f"layer_{layer_index:02d}.json"
    )

    layer_summary = {
        "layer": layer_index,
        "scale": SCALE,
        "results": layer_results
    }

    with open(
        layer_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            layer_summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    all_results.append(
        layer_summary
    )

    print(
        f"Saved: {layer_file}"
    )

    torch.cuda.empty_cache()


summary = []

for layer_data in all_results:

    category_scores = {}

    for result in layer_data["results"]:

        category = result["category"]

        if category not in category_scores:
            category_scores[category] = []

        category_scores[category].append(
            result["delta_nll"]
        )

    row = {
        "layer": layer_data["layer"]
    }

    for category, scores in category_scores.items():

        row[category] = (
            sum(scores) / len(scores)
        )

    row["overall"] = (
        sum(
            result["delta_nll"]
            for result in layer_data["results"]
        )
        / len(layer_data["results"])
    )

    summary.append(row)


summary_file = os.path.join(
    RESULTS_DIR,
    "sensitivity_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"\nBaseline time: "
    f"{baseline_time / 60:.2f} minutes"
)

print(
    f"Results directory: "
    f"{RESULTS_DIR}"
)

print(
    f"Summary: "
    f"{summary_file}"
)

print("\nLayer sensitivity:")

categories = list(benchmark.keys())

header = (
    f"{'Layer':<8}"
    + "".join(
        f"{category:<15}"
        for category in categories
    )
    + f"{'Overall':<15}"
)

print(header)
print("-" * len(header))

for row in summary:

    line = f"{row['layer']:<8}"

    for category in categories:

        value = row.get(
            category,
            0.0
        )

        line += f"{value:<15.6f}"

    line += f"{row['overall']:<15.6f}"

    print(line)

print(
    "\nOriginal model weights were never modified."
)
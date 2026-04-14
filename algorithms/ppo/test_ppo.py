import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from transformers import GPT2LMHeadModel, AutoTokenizer
from common.utils import get_device
import torch
import json


def generate_responses(model, tokenizer, prompts, device):
    """Generate responses for a list of formatted prompts."""
    model.eval()
    tokenizer.padding_side = 'left'
    encoded = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []
    prompt_len = encoded['input_ids'].shape[1]
    for i in range(len(prompts)):
        response = tokenizer.decode(output[i][prompt_len:], skip_special_tokens=True)
        responses.append(response)

    tokenizer.padding_side = 'right'
    return responses


def main():
    device = get_device()

    # --- Load template and prompts ---
    with open('prompts/ppo_prompt_template.md', 'r') as f:
        template = f.read()
    with open('data/ppo_prompts.json', 'r') as f:
        prompt_data = json.load(f)

    sample_prompts = [p['prompt'] for p in prompt_data[:10]]
    formatted_prompts = [template.format(prompt=p) for p in sample_prompts]

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # --- Base model (untrained) ---
    print("Generating with base model...")
    base_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    base_responses = generate_responses(base_model, tokenizer, formatted_prompts, device)

    # --- PPO-trained model ---
    print("Generating with PPO-trained model...")
    ppo_model = GPT2LMHeadModel.from_pretrained('checkpoints/ppo_policy').to(device)
    ppo_responses = generate_responses(ppo_model, tokenizer, formatted_prompts, device)

    # --- Save results ---
    results = []
    for i in range(len(sample_prompts)):
        results.append({
            'prompt': sample_prompts[i],
            'base_response': base_responses[i],
            'ppo_response': ppo_responses[i],
        })

    with open('results/ppo_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    # --- Print side by side ---
    for r in results:
        print(f"\nPrompt: {r['prompt']}")
        print(f"  Base: {r['base_response'][:200]}")
        print(f"  PPO:  {r['ppo_response'][:200]}")

    print(f"\nSaved comparison to results/ppo_comparison.json")


if __name__ == '__main__':
    main()

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from common.utils import get_device
from algorithms.dpo.dpo_trainer import DPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Online DPO with Claude-as-Judge")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_gen_len", type=int, default=64)
    parser.add_argument("--claude_model", type=str, default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    device = get_device()
    sft_path = "checkpoints/sft"

    # Policy model (trainable) — initialized from SFT checkpoint
    policy_model = GPT2LMHeadModel.from_pretrained(sft_path).to(device)

    # Reference model (frozen) — same SFT checkpoint
    ref_model = GPT2LMHeadModel.from_pretrained(sft_path).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    with open("data/ppo_prompts.json") as f:
        prompt_data = json.load(f)
    prompts = [item["prompt"] for item in prompt_data]

    trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        device=device,
        beta=args.beta,
        lr=args.lr,
        max_gen_len=args.max_gen_len,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        group_size=args.group_size,
        claude_model=args.claude_model,
    )
    trainer.train(prompts)


if __name__ == "__main__":
    main()

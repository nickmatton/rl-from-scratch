import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from transformers import GPT2LMHeadModel, AutoTokenizer
from common.utils import get_device
from models.reward_model import RewardModel
import torch
from algorithms.grpo.grpo_trainer import GRPOTrainer
import json

def main():
    model_name="gpt2"
    device = get_device()
    policy_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ref_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    reward_model = RewardModel(model_name).to(device)
    reward_model.load_state_dict(
        torch.load("checkpoints/reward/reward_model.pt", map_location=device)
    )

    # --- Load Prompts ---
    with open('data/ppo_prompts.json') as f:
        prompt_data = json.load(f)
    prompts = [item['prompt'] for item in prompt_data]

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        policy_model,
        ref_model,
        reward_model,
        tokenizer,
        device,
        lr=1e-5,
        kl_coef=0.1,
        max_gen_len=64,
        batch_size=4,
        num_iterations=50,
        grpo_epochs=2
    )
    trainer.train(prompts)

if __name__ == "__main__":
    main()

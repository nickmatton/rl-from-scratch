from transformers import GPT2LMHeadModel, AutoTokenizer
from value_model import ValueModel
from reward_model import RewardModel
from ppo_trainer import PPOTrainer
from utils import get_device
import torch
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description='PPO Training')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_iterations', type=int, default=50)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--max_gen_len', type=int, default=64)
    args = parser.parse_args()

    device = get_device()

    # --- Load Models ---
    model_name = 'gpt2'
    sft_path = 'checkpoints/sft'
    policy_model = GPT2LMHeadModel.from_pretrained(sft_path).to(device)
    ref_model = GPT2LMHeadModel.from_pretrained(sft_path).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    value_model = ValueModel(sft_path).to(device)
    for param in value_model.backbone.parameters():
        param.requires_grad = False

    reward_model = RewardModel(model_name).to(device)
    reward_model.load_state_dict(torch.load('checkpoints/reward/reward_model.pt', map_location=device))
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Load Prompts ---
    with open('data/ppo_prompts.json') as f:
        prompt_data = json.load(f)
    prompts = [item['prompt'] for item in prompt_data]

    # --- Create Trainer and Run ---
    trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        ppo_epochs=args.ppo_epochs,
        kl_coef=args.kl_coef,
        clip_eps=args.clip_eps,
        max_gen_len=args.max_gen_len,
    )
    trainer.train(prompts)

if __name__ == '__main__':
    main()

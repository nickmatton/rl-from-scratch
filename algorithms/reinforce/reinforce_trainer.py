# reinforce_trainer.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer

from common.utils import extract_gen_logprobs

class ReinforceTrainer:
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        reward_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        lr: float = 1e-5,
        kl_coef: float = 0.1,
        gamma: float = 1.0,
        max_gen_len: int = 64,
        batch_size: int  = 8,
        num_iterations: int = 50,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.num_iterations = num_iterations

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=lr, weight_decay=0.01
        )

    @torch.no_grad()
    def generate_rollouts(self, prompts: list[str]) -> dict:
        self.policy_model.eval()
        self.ref_model.eval()
        self.reward_model.eval()

        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        padded_prompt_len = encoded["input_ids"].shape[1]

        gen_output = self.policy_model.generate(
            **encoded,
            max_new_tokens=self.max_gen_len,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        full_ids = gen_output.sequences
        total_len = full_ids.shape[1]
        gen_ids = full_ids[:, padded_prompt_len:]

        ios_eos = gen_ids == self.tokenizer.eos_token_id
        eos_cumsum = ios_eos.cumsum(dim=1)
        gen_mask = ((eos_cumsum == 0 | (ios_eos & eos_cumsum ==1))).long()
        full_mask = torch.cat([encoded["attention_mask"], gen_mask], dim=1)

        policy_logits = self.policy_model(
            input_ids=full_ids,
            attention_mask=full_mask
        ).logits
        policy_logprobs = extract_gen_logprobs(
            policy_logits, gen_ids, padded_prompt_len, total_len
        )

        ref_logits = self.ref_model(
            input_ids=full_ids,
            attention_mask=full_mask
        ).logits
        ref_logprobs = extract_gen_logprobs(
            ref_logits, gen_ids, padded_prompt_len, total_len
        )

        reward_scores = self.reward_model(
            input_ids=full_ids,
            attention_mask=full_mask
        )

        response_texts = [
            self.tokenizer.decode(gen_ids[i][gen_mask[i].bool()], skip_special_tokens=True)
            for i in range(gen_ids.shape[0])
        ]

        kl_per_token = self.kl_coef * (policy_logprobs - ref_logprobs)

        self.tokenizer.padding_side = 'right'

        return {
            "full_ids": full_ids,
            "full_mask": full_mask,
            "gen_ids": gen_ids,
            "gen_mask": gen_mask,
            "old_logprobs": policy_logprobs,  # (batch, gen_len)
            "kl_per_token": kl_per_token,     # (batch, gen_len)
            "reward_scores": reward_scores,   # (batch,)
            "prompt_texts": prompts,
            "response_texts": response_texts,
        }

    def compute_monte_carlo_returns(self, reward_scores, kl_per_token, gen_mask):
        """
        Monte Carlo returns: G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}

        Uses actual observed rewards from the complete rollout (no bootstrapping).
        Unbiased but high variance compared to TD methods like GAE in PPO.
        """
        batch_size, gen_len = gen_mask.shape
        rewards = -kl_per_token.clone()

        last_real_idx = gen_mask.sum(dim=1) - 1

        rewards[torch.arange(batch_size, device=self.device), last_real_idx] += reward_scores

        returns = torch.zeros_like(rewards)
        for i in range(batch_size):
            G = 0.0
            for t in reversed(range(gen_len)):
                if gen_mask[i, t] == 0:
                    continue
                G = rewards[i, t] + self.gamma * G
                returns[i, t] = G

        return returns
    
    def reinforce_step(self, rollouts: dict):
        """Loss = -mean(log_prob(a_t) * G_t)   (over all real tokens in batch)"""
        # Set model to train
        self.policy_model.train()

        # Extract data
        full_ids = rollouts["full_ids"]
        full_mask = rollouts["full_mask"]
        gen_ids = rollouts["gen_ids"]
        gen_mask = rollouts["gen_mask"]
        total_len = full_ids.shape[1]
        gen_len = gen_ids.shape[1]
        prompt_len = total_len - gen_len

        logits = self.policy_model(
            input_ids=full_ids,
            attention_mask=full_mask
        ).logits
        log_probs = extract_gen_logprobs(logits, gen_ids, prompt_len, total_len)

        returns = self.compute_monte_carlo_returns(
            rollouts["reward_scores"], rollouts["kl_per_token"], gen_mask
        )

        mask_bool = gen_mask.bool()
        returns_masked = returns[mask_bool]
        if returns_masked.numel() > 1:
            returns_normalized = (returns_masked - returns_masked.mean()) / (
                returns_masked.std() + 1e-8
            )
        else:
            returns_normalized = returns_masked

        log_probs_masked = log_probs[mask_bool]
        policy_loss = -(log_probs_masked * returns_normalized.detach()).mean()

        # Gradient Step
        self.optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Logging stats
        with torch.no_grad():
            kl = (rollouts["old_logprobs"][mask_bool] - log_probs[mask_bool].detach()).mean().item()
            avg_reward = rollouts["reward_scores"].mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "mean_kl": kl,
            "avg_reward": avg_reward,
        }
    
    def train(self, prompts: list[str]):
        with open("prompts/reinforce_prompt_template.md", "r") as f:
            template = f.read()

        formatted_prompts = [template.format(prompt=p) for p in prompts]

        for iteration in range(self.num_iterations):
            batch_indicies = torch.randint(0, len(formatted_prompts), (self.batch_size,))
            batch_prompts = [formatted_prompts[i] for i in batch_indicies]

            rollouts = self.generate_rollouts(batch_prompts)
            stats = self.reinforce_step(rollouts)

            print(
            f"Iter {iteration + 1}/{self.num_iterations} | "
            f"Loss: {stats['policy_loss']:.4f} | "
            f"KL: {stats['mean_kl']:.4f} | "
            f"Avg Reward: {stats['avg_reward']:.4f}"
            )

            if (iteration + 1) % 10 == 0:
                print(f"  Sample: {rollouts['response_texts'][0][:200]}")

        # Save trained policy
        self.policy_model.save_pretrained("checkpoints/reinforce")
        self.tokenizer.save_pretrained("checkpoints/reinforce")
        print("Saved policy model to checkpoints/reinforce")

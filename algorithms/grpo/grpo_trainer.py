import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer
from common.utils import extract_gen_logprobs

class GRPOTrainer:
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        reward_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        lr: float = 1e-5,
        kl_coef: float = 0.1,
        clip_eps: float = 0.2,
        max_gen_len: int = 64,
        batch_size: int  = 8,
        num_iterations: int = 50,
        group_size: int = 6,
        grpo_epochs: int = 4,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.lr = lr
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.max_gen_len = max_gen_len
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=lr, weight_decay=0.01
        )
        # Lets try using Muon
        # self.optimizer = torch.optim.Muon(
        #     self.policy_model.parameters(),
        #     lr=0.02,
        #     weight_decay=0.01,
        #     momentum=0.95
        # )
        self.batch_size = batch_size
        self.group_size = group_size
        self.grpo_epochs = grpo_epochs
        self.num_iterations = num_iterations

    @torch.no_grad()
    def generate_rollouts(self, prompts: list[str]):

        # init tokenizer
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        gen_output = self.policy_model.generate(
            **encoded,
            max_new_tokens=self.max_gen_len,
            num_return_sequences=self.group_size,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True
        )

        full_ids = gen_output.sequences
        padded_prompt_len = encoded["input_ids"].shape[1]
        gen_ids = full_ids[:, padded_prompt_len:] # batch_size, gen_len
        total_len = full_ids.shape[1]
        B = len(prompts)
        group_indices = torch.arange(
            B, device=self.device
        ).repeat_interleave(self.group_size)

        is_eos = gen_ids == self.tokenizer.eos_token_id # batch_size, gen_len
        eos_cumsum = is_eos.cumsum(dim=1)
        gen_mask = ((eos_cumsum == 0) | (is_eos & (eos_cumsum == 1))).long()

        prompt_mask = encoded["attention_mask"].repeat_interleave(self.group_size, dim=0)
        full_mask = torch.cat([prompt_mask, gen_mask], dim=1)

        policy_logits = self.policy_model(
            input_ids=full_ids,
            attention_mask=full_mask
        ).logits
        policy_logprobs = extract_gen_logprobs(policy_logits, gen_ids, padded_prompt_len, total_len)

        ref_logits = self.ref_model(
            input_ids=full_ids,
            attention_mask=full_mask
        ).logits
        ref_logprobs = extract_gen_logprobs(ref_logits, gen_ids, padded_prompt_len, total_len)

        reward_scores = self.reward_model(
            input_ids=full_ids,
            attention_mask=full_mask
        )

        response_texts = [ self.tokenizer.decode(
                gen_ids[i][gen_mask[i].bool()],
                skip_special_tokens=True
            ) for i in range(gen_ids.shape[0])
        ]

        return {
            "full_ids": full_ids,           # (B*G, total_len)
            "full_mask": full_mask,         # (B*G, total_len)
            "gen_ids": gen_ids,             # (B*G, gen_len)
            "gen_mask": gen_mask,           # (B*G, gen_len)
            "old_logprobs": policy_logprobs,# (B*G, gen_len)
            "ref_logprobs": ref_logprobs,   # (B*G, gen_len)
            "reward_scores": reward_scores, # (B*G,)
            "group_indices": group_indices, # (B*G,)
            "prompt_texts": prompts,
            "response_texts": response_texts,
        }
    

    def compute_group_advantages(self, reward_scores, group_indices, gen_mask):
        """
        For each group, calc mean and std reward
        Advantages are how much better or worse a prompt did than average
        """
        num_groups = group_indices.max().item() + 1
        advantages = torch.zeros_like(reward_scores)

        for g in range(num_groups):
            mask = group_indices == g
            group_rewards = reward_scores[mask]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std()
            advantages[mask] = (group_rewards - group_mean) / (group_std + 1e-8)

        per_token_advantages = advantages.unsqueeze(1) * gen_mask.float()

        return per_token_advantages
    
    def grpo_step(self, rollouts: dict):
        self.policy_model.train()

        advantages = self.compute_group_advantages(
            rollouts["reward_scores"],
            rollouts["group_indices"],
            rollouts["gen_mask"]
        )
        
        full_ids = rollouts["full_ids"]
        full_mask = rollouts["full_mask"]
        gen_ids = rollouts["gen_ids"]
        gen_mask = rollouts["gen_mask"]
        old_logprobs = rollouts["old_logprobs"]
        ref_logprobs = rollouts["ref_logprobs"]
        total_len = full_ids.shape[1]
        gen_len =  gen_ids.shape[1]
        prompt_len = total_len - gen_len

        total_policy_loss = 0
        total_kl = 0
        for epoch in range(self.grpo_epochs):
            curr_logits = self.policy_model(
                input_ids=full_ids,
                attention_mask=full_mask
            ).logits
            curr_logprobs = extract_gen_logprobs(
                curr_logits, gen_ids, prompt_len, total_len
            )

            mask_bool = gen_mask.bool()
            adv_masked = advantages[mask_bool]
            adv_normalized = (adv_masked - adv_masked.mean()) / (adv_masked.std() + 1e-8)

            log_ratio = curr_logprobs[mask_bool] - old_logprobs[mask_bool]
            ratio = torch.exp(torch.clamp(log_ratio, -2.0, 2.0))

            surr1 = ratio * adv_normalized
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_normalized
            policy_loss = -torch.min(surr1, surr2).mean()

            kl_loss = self.kl_coef * (curr_logprobs[mask_bool] - ref_logprobs[mask_bool]).mean()

            loss = policy_loss + kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.policy_model.parameters(),max_norm=1.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_kl += kl_loss.item()

        return {
            "policy_loss": total_policy_loss / self.grpo_epochs,
            "mean_kl": total_kl / self.grpo_epochs,
            "avg_reward": rollouts["reward_scores"].mean().item(),
        }
    
    def train(self, prompts: list[str]):

        with open("prompts/grpo_prompt_template.md", "r") as f:
            template = f.read()
        formatted_prompts = [
            template.format(prompt=p) for p in prompts
        ]

        for iteration in range(self.num_iterations):
            batch_indices = torch.randint(0, len(formatted_prompts), (self.batch_size,))
            batch_prompts = [formatted_prompts[i] for i in batch_indices]
            rollouts = self.generate_rollouts(batch_prompts)
            stats = self.grpo_step(rollouts)

            print(
                f"Iter {iteration + 1}/{self.num_iterations} | "
                f"Policy Loss: {stats['policy_loss']:.4f} | "
                f"KL: {stats['mean_kl']:.4f} | "
                f"Avg Reward: {stats['avg_reward']:.4f}"
            )


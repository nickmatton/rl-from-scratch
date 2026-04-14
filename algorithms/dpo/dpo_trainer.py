import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import re
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer
import anthropic


class DPOTrainer:
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        beta: float = 0.1,
        lr: float = 1e-6,
        max_gen_len: int = 64,
        batch_size: int = 4,
        num_iterations: int = 50,
        group_size: int = 4,
        claude_model: str = "claude-sonnet-4-20250514",
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.group_size = group_size
        self.claude_model = claude_model

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=lr, weight_decay=0.01
        )
        self.claude_client = anthropic.Anthropic()

    @torch.no_grad()
    def generate_rollouts(self, prompts: list[str]):
        """Generate group_size completions per prompt."""
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
            return_dict_in_generate=True,
        )

        full_ids = gen_output.sequences  # (B*G, total_len)
        prompt_len = encoded["input_ids"].shape[1]
        gen_ids = full_ids[:, prompt_len:]  # (B*G, gen_len)

        # Mask: keep tokens up to and including the first EOS
        is_eos = gen_ids == self.tokenizer.eos_token_id
        eos_cumsum = is_eos.cumsum(dim=1)
        gen_mask = ((eos_cumsum == 0) | (is_eos & (eos_cumsum == 1))).long()

        # Build full attention mask
        prompt_mask = encoded["attention_mask"].repeat_interleave(self.group_size, dim=0)
        full_mask = torch.cat([prompt_mask, gen_mask], dim=1)

        # Decode response texts for Claude ranking
        response_texts = [
            self.tokenizer.decode(
                gen_ids[i][gen_mask[i].bool()], skip_special_tokens=True
            )
            for i in range(gen_ids.shape[0])
        ]

        return {
            "full_ids": full_ids,       # (B*G, total_len)
            "full_mask": full_mask,     # (B*G, total_len)
            "gen_ids": gen_ids,         # (B*G, gen_len)
            "gen_mask": gen_mask,       # (B*G, gen_len)
            "prompt_len": prompt_len,
            "response_texts": response_texts,
        }

    def rank_with_claude(self, prompt: str, responses: list[str]) -> tuple[int, int]:
        """Ask Claude to pick the best and worst response from a group.

        Returns (best_idx, worst_idx) as indices into the responses list.
        """
        numbered = "\n\n".join(
            f"--- Response {i} ---\n{r}" for i, r in enumerate(responses)
        )

        ranking_prompt = (
            f'Given this prompt:\n"{prompt}"\n\n'
            f"Here are {len(responses)} candidate responses:\n\n"
            f"{numbered}\n\n"
            f"Which response is BEST and which is WORST? "
            f"Consider helpfulness, accuracy, and coherence.\n\n"
            f"Reply with EXACTLY this format (just the numbers, nothing else):\n"
            f"BEST: <index>\n"
            f"WORST: <index>"
        )

        message = self.claude_client.messages.create(
            model=self.claude_model,
            max_tokens=50,
            messages=[{"role": "user", "content": ranking_prompt}],
        )

        text = message.content[0].text.strip()
        best_match = re.search(r"BEST:\s*(\d+)", text)
        worst_match = re.search(r"WORST:\s*(\d+)", text)

        if not best_match or not worst_match:
            # Fallback: pick first and last
            return 0, len(responses) - 1

        best_idx = int(best_match.group(1))
        worst_idx = int(worst_match.group(1))

        # Clamp to valid range
        best_idx = max(0, min(best_idx, len(responses) - 1))
        worst_idx = max(0, min(worst_idx, len(responses) - 1))

        # If Claude picks the same for both, shift worst
        if best_idx == worst_idx:
            worst_idx = (best_idx + 1) % len(responses)

        return best_idx, worst_idx

    def get_response_logprobs(self, model, full_ids, full_mask, prompt_len):
        """Compute summed log-probs over response tokens only.

        Returns (batch_size,) sequence-level log-probabilities.
        """
        with torch.no_grad() if not model.training else torch.enable_grad():
            logits = model(input_ids=full_ids, attention_mask=full_mask).logits

        # logits[:, t, :] predicts token t+1
        logprobs_all = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = full_ids[:, 1:]

        per_token_logprobs = logprobs_all.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq_len-1)

        # Mask: only count response tokens (zero out prompt positions)
        response_mask = full_mask[:, 1:].clone()
        response_mask[:, : prompt_len - 1] = 0
        per_token_logprobs = per_token_logprobs * response_mask

        return per_token_logprobs.sum(dim=-1)  # (batch,)

    def dpo_step(self, rollouts: dict, rankings: list[tuple[int, int]]):
        """Build preference pairs from rankings and run one DPO optimization step."""
        self.policy_model.train()

        full_ids = rollouts["full_ids"]
        full_mask = rollouts["full_mask"]
        prompt_len = rollouts["prompt_len"]

        # Select chosen/rejected sequences based on Claude's rankings
        chosen_indices = []
        rejected_indices = []
        for prompt_idx, (best_local, worst_local) in enumerate(rankings):
            chosen_indices.append(prompt_idx * self.group_size + best_local)
            rejected_indices.append(prompt_idx * self.group_size + worst_local)

        chosen_ids = full_ids[chosen_indices]      # (B, total_len)
        chosen_mask = full_mask[chosen_indices]     # (B, total_len)
        rejected_ids = full_ids[rejected_indices]   # (B, total_len)
        rejected_mask = full_mask[rejected_indices] # (B, total_len)

        # Policy log-probs (with gradients)
        policy_chosen_logps = self.get_response_logprobs(
            self.policy_model, chosen_ids, chosen_mask, prompt_len
        )
        policy_rejected_logps = self.get_response_logprobs(
            self.policy_model, rejected_ids, rejected_mask, prompt_len
        )

        # Reference log-probs (no gradients)
        with torch.no_grad():
            ref_chosen_logps = self.get_response_logprobs(
                self.ref_model, chosen_ids, chosen_mask, prompt_len
            )
            ref_rejected_logps = self.get_response_logprobs(
                self.ref_model, rejected_ids, rejected_mask, prompt_len
            )

        # DPO loss
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Metrics
        chosen_rewards = (self.beta * chosen_logratios).detach()
        rejected_rewards = (self.beta * rejected_logratios).detach()

        return {
            "loss": loss.item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        }

    def train(self, prompts: list[str]):
        """Main online DPO training loop.

        Each iteration: generate k rollouts -> rank with Claude -> DPO step.
        """
        with open("prompts/dpo_prompt_template.md", "r") as f:
            template = f.read()
        formatted_prompts = [template.format(prompt=p) for p in prompts]

        for iteration in range(self.num_iterations):
            # Sample a batch of prompts
            batch_indices = torch.randint(0, len(formatted_prompts), (self.batch_size,))
            batch_prompts = [formatted_prompts[i] for i in batch_indices]
            raw_prompts = [prompts[i] for i in batch_indices]

            # 1. Generate k completions per prompt
            rollouts = self.generate_rollouts(batch_prompts)

            # 2. Have Claude rank each group
            rankings = []
            for p_idx in range(self.batch_size):
                start = p_idx * self.group_size
                end = start + self.group_size
                group_responses = rollouts["response_texts"][start:end]
                best, worst = self.rank_with_claude(raw_prompts[p_idx], group_responses)
                rankings.append((best, worst))

            # 3. Build preference pairs and run DPO step
            stats = self.dpo_step(rollouts, rankings)

            print(
                f"Iter {iteration + 1}/{self.num_iterations} | "
                f"Loss: {stats['loss']:.4f} | "
                f"Margin: {stats['reward_margin']:.4f} | "
                f"Acc: {stats['accuracy']:.4f}"
            )

        # Save checkpoint
        save_path = "checkpoints/dpo"
        os.makedirs(save_path, exist_ok=True)
        self.policy_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

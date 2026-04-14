# Online DPO with Claude-as-Judge

Direct Preference Optimization (DPO) eliminates the need for a separate reward model by training the policy directly on preference pairs. The key insight: the optimal policy under a Bradley-Terry preference model has a closed-form relationship to the reward function:

```
r(x, y) = β * log(π(y|x) / π_ref(y|x)) + constant
```

This lets us substitute implicit rewards directly into the preference loss, avoiding reward modeling and RL entirely.

**Our twist:** instead of training on a static preference dataset, we run *online* DPO. Each round, GPT-2 generates k completions per prompt, Claude ranks them via the Anthropic API, and we train on the resulting preference pairs. This gives us fresh, on-policy preference data every round — combining the simplicity of DPO with the exploration benefits of online RL methods like GRPO.

## Prerequisites

1. **An SFT checkpoint** — Run SFT first (`python algorithms/sft/sft.py`) to produce `checkpoints/sft/`.
2. **Anthropic API access** — Your Claude Code subscription provides API access. Install the SDK: `pip install anthropic`.
3. **Prompts** — We reuse `data/ppo_prompts.json`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Each DPO Round                    │
│                                                     │
│  1. Sample batch of prompts                         │
│  2. Generate k completions per prompt (GPT-2)       │
│  3. Send completions to Claude for ranking           │
│  4. Best → chosen, Worst → rejected                 │
│  5. Compute DPO loss on (chosen, rejected) pairs     │
│  6. Update policy                                   │
└─────────────────────────────────────────────────────┘
```

Compare this to GRPO which also generates multiple completions per prompt but uses a learned reward model to score them. Here, Claude *is* the reward signal.

## Step 1: Generating Rollouts

Like GRPO, we generate multiple completions per prompt. The generation logic mirrors `grpo_trainer.py` — left-pad prompts, generate with sampling, build masks:

```python
@torch.no_grad()
def generate_rollouts(self, prompts: list[str]):
    self.tokenizer.padding_side = "left"
    encoded = self.tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    ).to(self.device)

    gen_output = self.policy_model.generate(
        **encoded,
        max_new_tokens=self.max_gen_len,
        num_return_sequences=self.group_size,
        do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=self.tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )

    full_ids = gen_output.sequences
    prompt_len = encoded["input_ids"].shape[1]
    gen_ids = full_ids[:, prompt_len:]

    # Mask: keep tokens up to and including the first EOS
    is_eos = gen_ids == self.tokenizer.eos_token_id
    eos_cumsum = is_eos.cumsum(dim=1)
    gen_mask = ((eos_cumsum == 0) | (is_eos & (eos_cumsum == 1))).long()

    # Decode response texts for Claude to rank
    response_texts = [
        self.tokenizer.decode(gen_ids[i][gen_mask[i].bool()], skip_special_tokens=True)
        for i in range(gen_ids.shape[0])
    ]

    # Group indices: maps each sequence back to its source prompt
    group_indices = torch.arange(len(prompts), device=self.device).repeat_interleave(self.group_size)

    return {
        "full_ids": full_ids,
        "gen_ids": gen_ids,
        "gen_mask": gen_mask,
        "prompt_len": prompt_len,
        "group_indices": group_indices,
        "response_texts": response_texts,
    }
```

We don't need to compute log-probs during rollout generation (unlike PPO/GRPO which need old log-probs for importance sampling). DPO computes all log-probs during the training step.

## Step 2: Claude-as-Judge Ranking

This is the key differentiator. Instead of a learned reward model, we ask Claude to rank the k completions. For each prompt group, we send the completions to Claude and ask it to return the indices of the best and worst:

```python
def rank_with_claude(self, prompt: str, responses: list[str]) -> tuple[int, int]:
    """Ask Claude to pick the best and worst response. Returns (best_idx, worst_idx)."""
    numbered = "\n\n".join(
        f"--- Response {i} ---\n{r}" for i, r in enumerate(responses)
    )

    ranking_prompt = f"""Given this prompt:
"{prompt}"

Here are {len(responses)} candidate responses:

{numbered}

Which response is BEST and which is WORST? Consider helpfulness, accuracy, and coherence.

Reply with EXACTLY this format (just the numbers, nothing else):
BEST: <index>
WORST: <index>"""

    message = self.claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{"role": "user", "content": ranking_prompt}],
    )

    # Parse "BEST: X\nWORST: Y"
    text = message.content[0].text.strip()
    best_idx = int(text.split("BEST:")[1].split("\n")[0].strip())
    worst_idx = int(text.split("WORST:")[1].strip())
    return best_idx, worst_idx
```

We use `claude-sonnet-4-20250514` for cost efficiency — we're making many API calls per training round. The ranking task is straightforward enough that Sonnet handles it well.

## Step 3: Building Preference Pairs

After ranking, we select the best and worst completions from each group and tokenize them as preference pairs:

```python
def build_preference_pairs(self, rollouts, rankings):
    """Convert ranked rollouts into tokenized (chosen, rejected) pairs."""
    chosen_ids_list, rejected_ids_list = [], []

    for prompt_idx, (best_local, worst_local) in enumerate(rankings):
        best_global = prompt_idx * self.group_size + best_local
        worst_global = prompt_idx * self.group_size + worst_local

        full_ids = rollouts["full_ids"]
        gen_mask = rollouts["gen_mask"]
        prompt_len = rollouts["prompt_len"]

        # Full sequence = prompt + generation, masked properly
        chosen_ids_list.append(full_ids[best_global])
        rejected_ids_list.append(full_ids[worst_global])

    chosen_ids = torch.stack(chosen_ids_list)
    rejected_ids = torch.stack(rejected_ids_list)

    # Build attention masks from the generation masks
    prompt_mask = torch.ones(len(rankings), prompt_len, ...) # see full code
    ...

    return chosen_ids, chosen_mask, rejected_ids, rejected_mask
```

## Step 4: Sequence Log-Probabilities

DPO needs the total log-probability of each sequence under both the policy and reference models. We sum per-token log-probs over non-padding response tokens:

```python
def get_response_logprobs(self, model, full_ids, full_mask, prompt_len):
    """Compute summed log-probs over response tokens only."""
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids=full_ids, attention_mask=full_mask).logits

    # logits[:, t, :] predicts token t+1
    logprobs_all = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = full_ids[:, 1:]

    per_token_logprobs = logprobs_all.gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Only sum over response tokens (after prompt)
    response_mask = full_mask[:, 1:].clone()
    response_mask[:, :prompt_len - 1] = 0  # zero out prompt positions
    per_token_logprobs = per_token_logprobs * response_mask

    return per_token_logprobs.sum(dim=-1)
```

Key detail: we mask out the prompt tokens so we only measure how likely the *response* is, not the prompt.

## Step 5: The DPO Loss

The DPO loss applies the Bradley-Terry model to implicit rewards:

```
L_DPO = -log σ(β * [(log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))])
```

```python
def compute_dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
                     ref_chosen_logps, ref_rejected_logps):
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    logits = self.beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    # Metrics
    chosen_rewards = (self.beta * chosen_logratios).detach()
    rejected_rewards = (self.beta * rejected_logratios).detach()

    return loss, {
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
    }
```

Compare this to the reward model's Bradley-Terry loss in `models/train_reward.py`:
```python
loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

Identical structure — DPO just replaces learned rewards with implicit log-ratio rewards.

## Step 6: The Training Loop

Each iteration: generate rollouts → rank with Claude → build pairs → DPO step. This is the online loop that makes our approach different from vanilla DPO:

```python
def train(self, prompts: list[str]):
    with open("prompts/dpo_prompt_template.md", "r") as f:
        template = f.read()
    formatted = [template.format(prompt=p) for p in prompts]

    for iteration in range(self.num_iterations):
        # Sample a batch of prompts
        indices = torch.randint(0, len(formatted), (self.batch_size,))
        batch_prompts = [formatted[i] for i in indices]
        raw_prompts = [prompts[i] for i in indices]

        # 1. Generate k completions per prompt
        rollouts = self.generate_rollouts(batch_prompts)

        # 2. Have Claude rank them
        rankings = []
        for p_idx in range(self.batch_size):
            start = p_idx * self.group_size
            end = start + self.group_size
            group_responses = rollouts["response_texts"][start:end]
            best, worst = self.rank_with_claude(raw_prompts[p_idx], group_responses)
            rankings.append((best, worst))

        # 3. Build preference pairs and run DPO step
        stats = self.dpo_step(rollouts, rankings)

        print(f"Iter {iteration+1}/{self.num_iterations} | "
              f"Loss: {stats['loss']:.4f} | "
              f"Margin: {stats['reward_margin']:.4f} | "
              f"Acc: {stats['accuracy']:.4f}")
```

## Step 7: The DPOTrainer Class

The full trainer lives in `dpo_trainer.py`. See that file for the complete implementation. It follows the same patterns as `grpo_trainer.py`:
- `generate_rollouts()` — generate k completions per prompt
- `rank_with_claude()` — API call to rank completions
- `dpo_step()` — build pairs + compute loss + optimize
- `train()` — outer loop tying it all together

## Step 8: Entry Point

The training script `train_dpo.py` follows the repo's `train_[algo].py` pattern. It loads an SFT checkpoint, initializes the trainer, and kicks off training:

```python
python algorithms/dpo/train_dpo.py --batch_size 4 --group_size 4 --num_iterations 50
```

See `train_dpo.py` for the full implementation.

## Summary: Online DPO vs PPO vs GRPO

| Aspect | PPO | GRPO | Online DPO |
|--------|-----|------|------------|
| Reward signal | Learned reward model | Learned reward model | Claude API |
| Generation at train time | Yes | Yes (k per prompt) | Yes (k per prompt) |
| Preference pairs | No | No (group-relative) | Yes (best vs worst) |
| Value model | Yes | No | No |
| Loss function | Clipped surrogate | Clipped surrogate | Bradley-Terry on log-ratios |
| Complexity | High | Medium | Low (no reward model training) |
| Cost | Compute only | Compute only | Compute + API calls |

Online DPO trades API cost for eliminating the reward model entirely. Claude provides a stronger, more flexible reward signal than a small learned reward model — it can judge quality along dimensions (factuality, safety, style) that would each require a separate reward model to capture.

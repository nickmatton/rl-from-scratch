# Implementing GRPO (Group Relative Policy Optimization) from Scratch

A hands-on tutorial using GPT-2 Small (124M parameters). GRPO is the algorithm behind DeepSeek-R1 — a simpler alternative to PPO that eliminates the value function entirely.

---

## What You'll Build

By the end of this tutorial, you'll have implemented GRPO from scratch:

1. **`grpo_trainer.py`** — The core GRPO trainer class with rollout generation, group advantage computation, and the clipped policy gradient step
2. **`train_grpo.py`** — Entry point script that wires up the models and runs training
3. **`prompts/grpo_prompt_template.md`** — Prompt template for generation

---

## What is GRPO and Why Does it Exist?

### The Problem with PPO

PPO for language models requires four models in memory at once:

| Model | Role | Trainable? |
|-------|------|-----------|
| Policy model | Generates responses | Yes |
| Reference model | KL penalty baseline | No (frozen) |
| Reward model | Scores responses | No (frozen) |
| Value model | Estimates expected future reward | Yes |

The value model is expensive — it's typically the same size as the policy model, doubling your trainable parameters. It also introduces a whole extra source of instability: if the value estimates are bad, your advantages are bad, and training goes off the rails.

### GRPO's Key Insight

Instead of learning a value function to estimate "how good is this state?", GRPO answers a simpler question: **"how good is this completion compared to other completions for the same prompt?"**

The algorithm:
1. For each prompt, generate **G completions** (a "group")
2. Score each completion with the reward model
3. **Normalize rewards within each group**: subtract the group mean, divide by the group std
4. Use these normalized rewards as advantages in a PPO-style clipped objective

This is the "group relative" part — advantages are always relative to the group, not to a learned baseline. The key simplifications:

- **No value model** — saves ~50% of trainable parameters
- **No GAE computation** — no need for per-token temporal difference errors
- **Sequence-level advantages** — every token in a completion gets the same advantage (the normalized group reward)
- **Simpler to implement and debug** — fewer moving parts

### The Math

Given a prompt $x$ and a group of $G$ completions $\{o_1, ..., o_G\}$ with rewards $\{r_1, ..., r_G\}$:

**Group normalization:**
$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G) + \epsilon}$$

**Clipped objective (same as PPO):**
$$\mathcal{L}_{\text{policy}} = -\frac{1}{|M|}\sum_{t \in M} \min\left(\rho_t \hat{A}, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}\right)$$

where $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ is the probability ratio and $M$ is the set of non-padding tokens.

**KL penalty (additive in the loss):**
$$\mathcal{L}_{\text{KL}} = \beta \cdot \frac{1}{|M|}\sum_{t \in M} \left(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{ref}}(a_t|s_t)\right)$$

**Total loss:**
$$\mathcal{L} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{KL}}$$

Note: the KL penalty here is an **additive loss term**, not per-token reward shaping like PPO does. This follows the GRPO paper's formulation more closely.

---

## Prerequisites

This tutorial assumes you've already completed Steps 1 and 2 from the RLHF tutorial:
- A trained **reward model** saved at `checkpoints/reward/reward_model.pt`
- Familiarity with the existing codebase patterns (`common/utils.py`, `models/reward_model.py`)

You should also be familiar with:
- The REINFORCE trainer at `algorithms/reinforce/reinforce_trainer.py` — GRPO reuses many of its patterns
- The PPO trainer at `algorithms/rlhf/ppo_trainer.py` — GRPO uses PPO's clipped objective but replaces GAE with group normalization

---

## Step 1: Create the Prompt Template

**File:** `prompts/grpo_prompt_template.md`

This is identical to the REINFORCE and PPO templates. Create a file with:

```
### Instruction: {prompt}
### Response:
```

This template gets formatted with each prompt string during training. The `{prompt}` placeholder is replaced via Python's `str.format()`.

---

## Step 2: Create the Entry Point Script

**File:** `algorithms/grpo/train_grpo.py`

Follow the exact same pattern as `algorithms/reinforce/train_reinforce.py`. This script wires up all the components and kicks off training.

### 2.1 — Imports and Path Setup

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
```

This path insertion is needed because our modules live at the repo root (`common/`, `models/`), but we run the script from inside `algorithms/grpo/`. Every training script in the project does this.

Import the same set of dependencies as `train_reinforce.py`:
- `torch`
- `GPT2LMHeadModel` and `AutoTokenizer` from transformers
- `get_device` from `common.utils`
- `RewardModel` from `models.reward_model`
- Your `GRPOTrainer` (which you'll create in Step 3)

### 2.2 — Load the Policy Model

```python
policy_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
```

This is the model we're training. It starts as base GPT-2 and gets optimized by GRPO to produce higher-reward responses.

### 2.3 — Load the Reference Model (Frozen)

```python
ref_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
```

The reference model is a frozen copy of the starting policy. It never changes during training. We use it to compute a KL divergence penalty that prevents the policy from diverging too far from reasonable language.

**Why freeze it?** If we didn't penalize divergence from the reference, the policy would quickly learn to output degenerate text that hacks the reward model (reward hacking). The KL penalty acts as an anchor.

### 2.4 — Load the Reward Model (Frozen)

```python
reward_model = RewardModel(model_name).to(device)
reward_model.load_state_dict(
    torch.load("checkpoints/reward/reward_model.pt", map_location=device)
)
reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False
```

This is the reward model trained in the RLHF tutorial's Step 2. It takes a full sequence (prompt + response) and outputs a scalar score. Higher scores = better responses.

**Note:** Unlike PPO, GRPO does **not** need a value model. This is the main simplification — we go from 4 models to 3.

### 2.5 — Tokenizer Setup

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

GPT-2 doesn't have a pad token by default, so we reuse the EOS token. This is standard across all our training scripts.

### 2.6 — Define Prompts and Instantiate Trainer

Use the same prompts list as REINFORCE. Instantiate `GRPOTrainer` with these hyperparameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lr` | `1e-5` | Learning rate for AdamW |
| `group_size` | `4` | Number of completions per prompt (G) |
| `kl_coef` | `0.1` | Weight of KL penalty in loss |
| `clip_eps` | `0.2` | PPO clipping range |
| `max_gen_len` | `64` | Max tokens to generate |
| `batch_size` | `4` | Number of unique prompts per iteration |
| `num_iterations` | `50` | Total training iterations |
| `grpo_epochs` | `2` | Times to reuse each set of rollouts |

**Important:** `batch_size` here means the number of **unique prompts**. The actual number of completions per iteration is `batch_size * group_size` = 16. This is a key difference from REINFORCE/PPO where `batch_size` is the total number of completions.

Call `trainer.train(prompts)` to start training.

---

## Step 3: Implement the GRPO Trainer

**File:** `algorithms/grpo/grpo_trainer.py`

This is the core of the tutorial. We'll build the `GRPOTrainer` class step by step.

### 3.1 — Imports and Class Setup

Use the same imports as `reinforce_trainer.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer
```

### 3.2 — `__init__`: Constructor

The constructor takes the same models as REINFORCE (policy, ref, reward, tokenizer, device) plus GRPO-specific hyperparameters.

**Key difference from PPO:** only one optimizer, for the policy model only. No value model, no value optimizer.

```python
self.optimizer = torch.optim.AdamW(
    self.policy_model.parameters(), lr=lr, weight_decay=0.01
)
```

Store all hyperparameters as instance attributes: `group_size`, `kl_coef`, `clip_eps`, `max_gen_len`, `batch_size`, `num_iterations`, `grpo_epochs`.

### 3.3 — `generate_rollouts`: The Rollout Generation Step

This is the most involved method. Decorate it with `@torch.no_grad()` since we're only generating data, not training.

#### 3.3.1 — Tokenize and Generate with `num_return_sequences`

Set models to eval mode, then left-pad and tokenize the prompts (no expansion needed):

```python
self.tokenizer.padding_side = "left"
encoded = self.tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
).to(self.device)
```

**Why left-pad?** GPT-2 is a causal (left-to-right) model. Left-padding ensures all real tokens are contiguous on the right side, so generation can proceed naturally from the last real token. Right-padding would put pads in the middle of the sequence for shorter prompts, confusing the model.

Generate G completions per prompt using `num_return_sequences`:

```python
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
```

Under the hood, HuggingFace calls `input_ids.repeat_interleave(num_return_sequences, dim=0)` — so the output shape is `(B*G, total_len)`, with every G consecutive rows belonging to the same prompt.

Extract `full_ids` and `gen_ids`:

```python
full_ids = gen_output.sequences              # (B*G, total_len)
padded_prompt_len = encoded["input_ids"].shape[1]
gen_ids = full_ids[:, padded_prompt_len:]     # (B*G, gen_len)
```

#### 3.3.2 — Build Group Indices

We need a `group_indices` tensor that maps each completion back to its prompt group. Since `num_return_sequences` repeats each input G times contiguously, the mapping is straightforward:

```python
B = len(prompts)
G = self.group_size
group_indices = torch.arange(B, device=self.device).repeat_interleave(G)
# For B=2, G=4: tensor([0, 0, 0, 0, 1, 1, 1, 1])
```

This will be used later in `compute_group_advantages` to identify which completions belong to the same group.

#### 3.3.3 — Build the Generation Attention Mask

We need to mask out everything after the first EOS token. This is the same `eos_cumsum` pattern used in REINFORCE and PPO:

```python
is_eos = gen_ids == self.tokenizer.eos_token_id
eos_cumsum = is_eos.cumsum(dim=1)
gen_mask = ((eos_cumsum == 0) | (is_eos & (eos_cumsum == 1))).long()
```

How this works:
- `eos_cumsum == 0` is True for all tokens before the first EOS
- `is_eos & (eos_cumsum == 1)` is True for exactly the first EOS token
- Everything after the first EOS gets masked out (0 in gen_mask)

One wrinkle from using `num_return_sequences`: `encoded["attention_mask"]` has shape `(B, prompt_len)` but `gen_mask` has shape `(B*G, gen_len)`. We need to expand the prompt mask to match before concatenating:

```python
prompt_mask = encoded["attention_mask"].repeat_interleave(self.group_size, dim=0)  # (B*G, prompt_len)
full_mask = torch.cat([prompt_mask, gen_mask], dim=1)
```

#### 3.3.4 — Compute Policy and Reference Log-Probabilities

Run a forward pass through both the policy and reference models, then extract log-probs for the generated tokens. Use the `_extract_gen_logprobs` helper (same as in REINFORCE):

```python
policy_logits = self.policy_model(
    input_ids=full_ids, attention_mask=full_mask
).logits
policy_logprobs = extract_gen_logprobs(
    policy_logits, gen_ids, padded_prompt_len, total_len
)
```

Do the same for the reference model to get `ref_logprobs`.

**Important:** We need `ref_logprobs` stored in the rollouts for the KL penalty computation during `grpo_step`. In REINFORCE, the KL was computed here in rollout generation. In GRPO, we compute it during the optimization step because we need the **current** policy's logprobs (which change across GRPO epochs), not the old ones.

#### 3.3.5 — Compute Reward Scores

```python
reward_scores = self.reward_model(
    input_ids=full_ids, attention_mask=full_mask
)  # (B*G,)
```

This gives us one scalar reward per completion. These will be normalized within groups in `compute_group_advantages`.

#### 3.3.6 — Decode and Pack Rollouts

Decode response texts for logging, reset the tokenizer padding side, and return the rollouts dictionary:

```python
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
```

**Key difference from REINFORCE rollouts:** We store `ref_logprobs` and `group_indices` separately, and we do **not** compute `kl_per_token` here. The KL is computed fresh during optimization.

### 3.4 — Log-Probability Extraction via `extract_gen_logprobs`

This is a shared utility in `common/utils.py`, also used by REINFORCE and PPO. Import it:

```python
from common.utils import extract_gen_logprobs
```

Then call it as a free function (not a method):

```python
policy_logprobs = extract_gen_logprobs(policy_logits, gen_ids, padded_prompt_len, total_len)
```

**How it works:** In a causal language model, logits at position $t$ predict the token at position $t+1$. So to get the log-probability of each generated token, we need the logits at positions `[prompt_len-1, prompt_len, ..., total_len-2]`, which predict tokens at positions `[prompt_len, prompt_len+1, ..., total_len-1]`. The function applies `log_softmax`, slices to the generation range, and `.gather()`s the log-probability of each actually-generated token.

### 3.5 — `compute_group_advantages`: The Core GRPO Innovation

This method replaces PPO's GAE (Generalized Advantage Estimation) with a much simpler computation.

**Input:**
- `reward_scores`: shape `(B*G,)` — one reward per completion
- `group_indices`: shape `(B*G,)` — maps each completion to its group (e.g., `[0,0,0,0,1,1,1,1]`)
- `gen_mask`: shape `(B*G, gen_len)` — which tokens are real (not padding)

**Algorithm:**

For each group $g$ of completions that share the same prompt:
1. Gather the rewards for that group
2. Compute the group mean and standard deviation
3. Normalize: $\hat{A}_i = (r_i - \mu_g) / (\sigma_g + \epsilon)$

```python
num_groups = group_indices.max().item() + 1   # B (number of unique prompts)
advantages = torch.zeros_like(reward_scores)  # (B*G,)

for g in range(num_groups):
    mask = group_indices == g                  # (B*G,) bool — True for G entries in this group
    group_rewards = reward_scores[mask]        # (G,)   — rewards for this group's completions
    group_mean = group_rewards.mean()          # ()     — scalar
    group_std = group_rewards.std()            # ()     — scalar
    advantages[mask] = (group_rewards - group_mean) / (group_std + 1e-8)  # (G,)
# advantages is now (B*G,) — one normalized advantage per completion
```

Then broadcast the per-sequence advantage to every token in that sequence:

```python
per_token_advantages = advantages.unsqueeze(1) * gen_mask.float()
# advantages:          (B*G,)                — one value per completion
# .unsqueeze(1):       (B*G, 1)              — add token dimension
# gen_mask.float():    (B*G, gen_len)        — 1.0 for real tokens, 0.0 for padding
# result:              (B*G, gen_len)        — same advantage at every real token, 0 at padding
```

**Why this works:** The normalized advantages tell us, within each group, which completions were above average (positive advantage) and which were below (negative advantage). The policy is pushed to make good completions more likely and bad completions less likely — all relative to the group.

**Why `1e-8` in the denominator?** If all G completions in a group happen to get the exact same reward, the standard deviation is 0, and we'd divide by zero. The epsilon prevents this. In practice, this is rare with sampled generation.

**Comparison to PPO's GAE:**
- PPO assigns **different advantages to different tokens** within a single completion, using temporal difference errors and a learned value function
- GRPO assigns **the same advantage to every token** in a completion — the normalized group reward
- This is less granular but much simpler and doesn't require a value model

### 3.6 — `grpo_step`: The Optimization Step

This method takes the rollouts, computes advantages, and runs multiple epochs of gradient descent.

#### 3.6.1 — Setup

Set the policy to train mode and compute group advantages:

```python
self.policy_model.train()

advantages = self.compute_group_advantages(
    rollouts["reward_scores"],
    rollouts["group_indices"],
    rollouts["gen_mask"],
)
```

Unpack the rollouts:

```python
full_ids = rollouts["full_ids"]
full_mask = rollouts["full_mask"]
gen_ids = rollouts["gen_ids"]
gen_mask = rollouts["gen_mask"]
old_logprobs = rollouts["old_logprobs"]   # policy logprobs from rollout time
ref_logprobs = rollouts["ref_logprobs"]   # frozen reference model logprobs
total_len = full_ids.shape[1]
gen_len = gen_ids.shape[1]
prompt_len = total_len - gen_len
```

`old_logprobs` are the policy's log-probabilities recorded during `generate_rollouts` — they represent $\pi_{\text{old}}$ in the probability ratio. As the policy updates across GRPO epochs, `curr_logprobs` will diverge from `old_logprobs`, and the ratio measures that divergence.

#### 3.6.2 — The GRPO Epoch Loop

Like PPO, GRPO reuses rollouts for multiple optimization epochs. This is more sample-efficient than REINFORCE (which uses each rollout exactly once).

```python
for epoch in range(self.grpo_epochs):
```

Within each epoch:

**a) Forward pass on current policy:**

```python
curr_logits = self.policy_model(
    input_ids=full_ids, attention_mask=full_mask
).logits
curr_logprobs = extract_gen_logprobs(
    curr_logits, gen_ids, prompt_len, total_len
)
```

Note that `curr_logprobs` changes each epoch as the policy updates. This is why we can't just precompute it once.

**b) Normalize advantages:**

```python
mask_bool = gen_mask.bool()
adv_masked = advantages[mask_bool]
adv_normalized = (adv_masked - adv_masked.mean()) / (adv_masked.std() + 1e-8)
```

This is a second normalization on top of the group normalization. It standardizes advantages across the entire batch for more stable gradient steps. This is the same pattern used in PPO and REINFORCE.

**c) Compute probability ratios:**

```python
log_ratio = curr_logprobs[mask_bool] - old_logprobs[mask_bool]
ratio = torch.exp(torch.clamp(log_ratio, -2.0, 2.0))
```

The ratio $\rho_t = \pi_\theta(a_t) / \pi_{\text{old}}(a_t)$ measures how much the current policy differs from the policy that generated the rollouts. We clamp the log-ratio to `[-2, 2]` before exponentiating to prevent numerical instability (same as PPO).

**d) Clipped surrogate loss (identical to PPO):**

```python
surr1 = ratio * adv_normalized
surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_normalized
policy_loss = -torch.min(surr1, surr2).mean()
```

The clipping prevents the policy from changing too much in a single step. If the ratio goes outside `[1-eps, 1+eps]`, the gradient is zeroed out. This is the core stability mechanism of PPO/GRPO.

**e) KL penalty:**

```python
kl_loss = self.kl_coef * (curr_logprobs[mask_bool] - ref_logprobs[mask_bool]).mean()
```

This penalizes the **current** policy (not the old rollout policy) for diverging from the frozen reference model. Note that we use `curr_logprobs` — this is why we needed to store `ref_logprobs` in the rollouts rather than precomputing the KL.

**Why additive KL and not reward-shaping KL?** In PPO, the KL penalty is typically added as a per-token reward, which flows through GAE into per-token advantages. Since GRPO doesn't have per-token rewards or GAE, we add it directly to the loss. This is simpler and follows the GRPO paper's formulation.

**f) Gradient step:**

```python
loss = policy_loss + kl_loss

self.optimizer.zero_grad()
loss.backward()
clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
self.optimizer.step()
```

Gradient clipping at `max_norm=1.0` prevents exploding gradients, same as REINFORCE and PPO.

#### 3.6.3 — Return Metrics

Track and average across epochs, then return:

```python
return {
    "policy_loss": total_policy_loss / n,
    "mean_kl": total_kl / n,
    "avg_reward": rollouts["reward_scores"].mean().item(),
}
```

### 3.7 — `train`: The Main Training Loop

This follows the exact same pattern as REINFORCE and PPO:

1. Load the prompt template from `prompts/grpo_prompt_template.md`
2. Format all prompts with the template
3. For each iteration:
   a. Sample a random batch of `batch_size` prompts
   b. Call `generate_rollouts(batch_prompts)` — this generates `batch_size * group_size` completions
   c. Call `grpo_step(rollouts)` — this runs the GRPO optimization
   d. Log metrics
   e. Every 10 iterations, print a sample response
4. Save the trained policy to `checkpoints/grpo/`

```python
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
```

---

## Step 4: Run Training

```bash
python algorithms/grpo/train_grpo.py
```

**Prerequisites:** You need a trained reward model at `checkpoints/reward/reward_model.pt`. If you don't have one, run the reward model training from the RLHF tutorial first.

### Expected Output

```
Iter 1/50 | Policy Loss: 0.0023 | KL: -0.0012 | Avg Reward: -0.1234
Iter 2/50 | Policy Loss: 0.0019 | KL: 0.0034 | Avg Reward: -0.0987
...
Iter 10/50 | Policy Loss: -0.0045 | KL: 0.0156 | Avg Reward: 0.1234
  Sample: The ocean is a vast body of water that covers...
...
Iter 50/50 | Policy Loss: -0.0123 | KL: 0.0432 | Avg Reward: 0.4567
  Sample: Gravity is the force that attracts objects...
Saved policy model to checkpoints/grpo
```

### What to Look For

- **Avg Reward should increase** over training — the policy is learning to generate higher-reward responses
- **KL should stay moderate** (roughly 0.01–0.1) — if it spikes, the policy is diverging too fast from the reference. Try lowering the learning rate or increasing `kl_coef`
- **Policy Loss should decrease** (become more negative) — the policy is getting better at producing high-advantage completions

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Reward doesn't improve | Group size too small | Increase `group_size` from 4 to 8 |
| KL explodes | Learning rate too high | Reduce `lr` to `5e-6` |
| NaN losses | Log-ratio overflow | Verify the `torch.clamp(log_ratio, -2.0, 2.0)` is in place |
| OOM errors | Too many completions | Reduce `batch_size` or `group_size` |
| All group advantages are 0 | All completions get the same reward | Check that your reward model produces varied scores |

---

## Comparison: GRPO vs PPO vs REINFORCE

| Aspect | REINFORCE | PPO | GRPO |
|--------|-----------|-----|------|
| Models needed | 3 (policy, ref, reward) | 4 (+ value model) | 3 (policy, ref, reward) |
| Advantage computation | Monte Carlo returns | GAE (per-token, learned baseline) | Group normalization (per-sequence) |
| Rollout reuse | 1 epoch | Multiple epochs | Multiple epochs |
| Clipped objective | No | Yes | Yes |
| KL penalty | Per-token reward shaping | Per-token reward shaping | Additive loss term |
| Variance | High | Low (GAE + value baseline) | Medium (group normalization) |
| Complexity | Simple | Complex | Medium |

### When to Use GRPO

- When you want PPO-level stability without the complexity of a value function
- When memory is constrained (no value model = fewer parameters)
- When your reward signal is at the sequence level (not token-level), which is the common case for RLHF
- When you can afford to generate multiple completions per prompt (the group size is a tradeoff between variance reduction and compute)

---

## Summary of Files to Create

1. **`prompts/grpo_prompt_template.md`** — Two lines: `### Instruction: {prompt}` and `### Response:`

2. **`algorithms/grpo/train_grpo.py`** — Entry point. Load policy (trainable), reference (frozen), reward model (frozen), tokenizer. No value model. Instantiate `GRPOTrainer` and call `train()`.

3. **`algorithms/grpo/grpo_trainer.py`** — Core class with:
   - `__init__` — Store models, hyperparams, single AdamW optimizer (policy only)
   - `generate_rollouts` — Batch generate G completions per prompt via `num_return_sequences`, compute policy/ref logprobs and rewards, track group_indices
   - Uses `extract_gen_logprobs` from `common/utils.py` (shared with REINFORCE and PPO)
   - `compute_group_advantages` — Normalize rewards within each group, broadcast to per-token
   - `grpo_step` — Clipped surrogate loss + additive KL penalty, multi-epoch
   - `train` — Sample batches, generate rollouts, optimize, log, save
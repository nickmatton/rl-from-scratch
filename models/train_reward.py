# train_reward.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset

from models.preference_dataset import PreferenceDataset
from models.reward_model import RewardModel
from common.utils import get_device


def train_reward_model(num_epochs=2, batch_size=16, learning_rate=5e-5, max_samples=20000, resume=False):
    model_name = "gpt2"
    device = get_device()

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = RewardModel(model_name=model_name)

    save_path = "checkpoints/reward"
    if resume:
        checkpoint_path = f"{save_path}/reward_model.pt"
        print(f"Resuming from checkpoint: {checkpoint_path}")
        # Load to CPU first to avoid MPS memory alignment errors with float16 tensors
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    model.to(device)

    # Load Anthropic hh-rlhf dataset from HuggingFace
    print(f"Loading Anthropic hh-rlhf dataset (using {max_samples} samples)...")
    hh_dataset = load_dataset("Anthropic/hh-rlhf", split=f"train[:{max_samples}]")
    dataset = PreferenceDataset(hh_dataset, tokenizer, use_template=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    print(f"Loaded {len(dataset)} preference pairs")

    # Setup training log
    os.makedirs(save_path, exist_ok=True)
    log_path = f"{save_path}/training_log.csv"
    if resume and os.path.exists(log_path):
        log_file = open(log_path, 'a', newline='')
    else:
        log_file = open(log_path, 'w', newline='')
        csv.writer(log_file).writerow(['epoch', 'loss', 'accuracy'])
    log_writer = csv.writer(log_file)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * num_epochs
    )

    # GradScaler for mixed-precision training: scales loss up before backward()
    # to prevent small gradients from underflowing to zero in float16,
    # then scales them back down before the optimizer step.
    scaler = torch.amp.GradScaler("mps")

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_pairs = 0

        for step, batch in enumerate(dataloader):
            chosen_ids = batch['chosen_ids'].to(device)
            chosen_mask = batch['chosen_mask'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)
            rejected_mask = batch['rejected_mask'].to(device)

            # autocast runs the forward pass in float16 for speed/memory,
            # while keeping master weights in float32 for accuracy
            with torch.amp.autocast("mps", dtype=torch.float16):
                chosen_rewards = model(chosen_ids, chosen_mask)
                rejected_rewards = model(rejected_ids, rejected_mask)

                # Bradley-Terry loss: -log(sigmoid(chosen_reward - rejected_reward))
                loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()      # scale loss up to prevent gradient underflow in float16
            scaler.unscale_(optimizer)           # scale gradients back to true values for clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)               # skips step if gradients overflowed (inf/nan)
            scaler.update()                      # adjust scale factor: decrease on overflow, increase when stable
            scheduler.step()

            total_loss += loss.item()
            total_correct += (chosen_rewards > rejected_rewards).sum().item()
            total_pairs += chosen_rewards.shape[0]

            if (step + 1) % 10 == 0:
                running_acc = total_correct / total_pairs
                running_loss = total_loss / (step + 1)
                print(f"  Step {step+1}/{len(dataloader)} | Loss: {running_loss:.4f} | Acc: {running_acc:.4f}")

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_pairs
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Log to CSV
        log_writer.writerow([epoch + 1, avg_loss, accuracy])
        log_file.flush()

    log_file.close()
    print(f"Training log saved to {log_path}")

    model_save_path = f"{save_path}/reward_model.pt"
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from saved checkpoint")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    args = parser.parse_args()
    train_reward_model(num_epochs=args.epochs, resume=args.resume)

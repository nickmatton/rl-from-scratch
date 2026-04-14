# sft.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from json import load
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from common.utils import get_device
from algorithms.sft.sft_dataset import SFTDataset

def train(model_name="gpt2", data_path="data/sft_data.json", output_dir="sft_model", epochs=20, batch_size=8, learning_rate=5e-5, log_file="checkpoints/sft/training_log.csv"):
    device = get_device()

    data = load(open(data_path, 'r'))
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    dataset = SFTDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    # Adam optimization is best explained IN A SINGLE SENTENCE: it adjusts the learning rate for each parameter based on the first and second moments of the gradients, allowing for efficient training of deep neural networks.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * epochs    
        
    # The scheduler will reduce the learning rate by a factor of 0.1 every third of the total training steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

    # Initialize log file with header
    with open(log_file, 'w') as f:
        f.write("epoch,loss,learning_rate\n")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100 # Mask out padding tokens in the loss calculation

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevents exploding gradients by clipping the norm of the gradients to a maximum value of 1.0    
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.2e}")
        
        # Write metrics to log file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{current_lr:.2e}\n")

    save_path = "checkpoints/sft"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    print(f"Training log saved to {log_file}")

if __name__ == "__main__":
    train()
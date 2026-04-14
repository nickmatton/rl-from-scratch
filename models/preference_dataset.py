# preference_dataset.py
import torch
from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    """
    Supports two data formats:
    1. Local JSON: list of dicts with 'prompt', 'chosen', 'rejected' keys
       (formatted with prompt_template.md)
    2. HuggingFace hh-rlhf: dicts with 'chosen', 'rejected' keys containing
       full conversation strings (used directly, no template)
    """
    def __init__(self, data, tokenizer, max_length=256, use_template=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_template = use_template

        if use_template:
            with open("prompts/prompt_template.md", 'r') as f:
                self.template = f.read()

        self.data = data

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text):
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(), encoded['attention_mask'].squeeze()

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.use_template:
            chosen_text = self.template.format(
                prompt=item['prompt'],
                completion=item['chosen'],
            ) + self.tokenizer.eos_token
            rejected_text = self.template.format(
                prompt=item['prompt'],
                completion=item['rejected'],
            ) + self.tokenizer.eos_token
        else:
            # HuggingFace hh-rlhf format: chosen/rejected are full conversations
            chosen_text = item['chosen'] + self.tokenizer.eos_token
            rejected_text = item['rejected'] + self.tokenizer.eos_token

        chosen_ids, chosen_mask = self._tokenize(chosen_text)
        rejected_ids, rejected_mask = self._tokenize(rejected_text)

        return {
            'chosen_ids': chosen_ids,
            'chosen_mask': chosen_mask,
            'rejected_ids': rejected_ids,
            'rejected_mask': rejected_mask,
        }

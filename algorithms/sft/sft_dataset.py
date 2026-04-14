# sft_dataset.py
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        prompt_path = "prompts/prompt_template.md"
        with open(prompt_path, 'r') as f:
            template = f.read()
            
        self.data = []
        for item in data:
            prompt = template.format(
                prompt=item['prompt'],
                completion=item['completion']
            ) + tokenizer.eos_token
            encoded_prompt = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.data.append(
                {
                    'input_ids': encoded_prompt['input_ids'].squeeze(),
                    'attention_mask': encoded_prompt['attention_mask'].squeeze()
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

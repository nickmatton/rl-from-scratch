# ppo_dataset.py
from torch.utils.data import Dataset

class PPODataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length=512):
        tempate_path = "prompts/ppo_prompt_template.md"
        with open(tempate_path, 'r') as f:
            template = f.read()

        self.data = []
        for prompt in prompts:
            prompt_text = template.format(prompt=prompt)

            # When batching, we pad each prompt to be the same length
            # And we track the original prompt length so we can separate it from the generated response later
            encoded_prompt = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            prompt_len = encoded_prompt['attention_mask'].sum().item()
            self.data.append(
                {
                    'input_ids': encoded_prompt['input_ids'].squeeze(),
                    'attention_mask': encoded_prompt['attention_mask'].squeeze(),
                    'prompt_len': prompt_len
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
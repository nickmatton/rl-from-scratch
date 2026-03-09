# utils.py
import torch
import torch.nn.functional as F

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def extract_gen_logprobs(logits, gen_ids, prompt_len, total_len):
    """Extract log-probabilities for generated tokens from model logits.

    logits[:, t, :] predicts token at position t+1, so we offset by -1.
    """
    logprobs_all = F.log_softmax(logits, dim=-1)
    logprobs_gen = logprobs_all[:, prompt_len - 1 : total_len - 1, :]
    return logprobs_gen.gather(
        dim=-1, index=gen_ids.unsqueeze(-1)
    ).squeeze(-1)


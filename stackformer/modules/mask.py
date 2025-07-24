# problem: Random mask and global mask
import torch

def casual_mask(Seq_len):
    causal_mask = torch.triu(torch.ones(Seq_len, Seq_len, dtype=torch.bool), diagonal=1)
    return causal_mask

def sliding_window(Seq_len, window_size):
    casual = torch.tril(torch.ones(Seq_len,Seq_len,dtype=bool))
    band = torch.triu(casual, diagonal=-(window_size-1))
    return ~band

def dilated_casual_mask(Seq_len, dilation):
    i = torch.arange(Seq_len).unsqueeze(1)
    j = torch.arange(Seq_len).unsqueeze(0)
    # causal and dilation condition
    mask = (i >= j) & ((i - j) % dilation == 0)
    return ~mask

def random_mask(Seq_len, num_random):
    mask = torch.zeros(Seq_len, Seq_len)
    for i in range(Seq_len):
        candidates = list(range(i))
        if len(candidates) == 0:
            continue
        random_mask = torch.randperm(len(candidates))[:min(num_random, len(candidates))]
        mask[i, torch.tensor([candidates[j] for j in random_mask])] = 1
    return ~mask

def global_mask(Seq_len, global_index):
    global_index_tensor = torch.tensor(global_index)
    mask = torch.zeros(Seq_len, Seq_len)
    for g in global_index:
        mask[g,:] = 1
    mask[:,global_index_tensor] = 1
    return ~mask
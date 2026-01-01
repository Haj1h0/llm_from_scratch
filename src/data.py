# src/data.py
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride): 
        assert max_length > 0, "max_length must be > 0"
        assert stride > 0, "stride must be > 0"

        self.input_ids = [] 
        self.target_ids = []  

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) >= max_length + 1, "토큰화된 입력의 개수는 적어도 max_length+1과 같아야 합니다."
        
        # sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1] 
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self): 
        return len(self.input_ids) 

    def __getitem__(self, idx): 
        return self.input_ids[idx], self.target_ids[idx]
    



def create_dataloader_v1(
    txt,
    tokenizer,
    batch_size=4, 
    max_length=256,  
    stride=128, 
    shuffle=True, 
    drop_last=True, 
    num_workers=0
):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  
        drop_last=drop_last,
        num_workers=num_workers 
    )
    return dataloader
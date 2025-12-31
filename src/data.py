# GPTDatasetV1 (cell 36) 


import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    # max_length : 모델이 한 번에 볼 수 있는 context size
    # stride: sliding window우가 한 번에 옆으로 건너뛰는(이동하는) 토큰의 개수
    # A token divided by the sliding window in the input_ids and target_ids list continues to be appended.
    def __init__(self, txt, tokenizer, max_length, stride): 
        self.input_ids = []  # x (batch의 입력)
        self.target_ids = []  # y (next-token target)

        # Tokenize the entire text
        # context window 구성을 위한 최소 토큰 길이를 검증하고, 미달 시 assert로 실행을 강제 중단
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "토큰화된 입력의 개수는 적어도 max_length+1과 같아야 합니다."
        
        # sliding window로 max_length size의 데이터를 stride 간격만큼 겹쳐가며 시퀀스를 분할
        # 이후 텐서로 변환해 리스트에 저장
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1] 
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Dataset size
    # input_ids 개수 == chunk 개수, 즉, 몇 개의 (x, y)쌍이 있는지 반환
    def __len__(self): 
        return len(self.input_ids) 

    # # 특정 index의 sample을 반환
    def __getitem__(self, idx): 
        return self.input_ids[idx], self.target_ids[idx]
    
# create_dataloader_v1 (cell 37) 
# dataloader :모델 학습에 쓸 데이터를 잘라서, 정해진 크기(batch)로, 반복적으로 모델에게 공급해주는 도구 
# dataloder 생성 시, shuffle은 순서나 패턴을 통째로 외우는 것(Overfitting)을 방지하기 위해 사용
# 학습(Train) 단계에서는 True로 설정해 매번 다른 순서로 학습, 평가(Test/Validation) 단계에서는 결과 비교를 위해 False
# 고로 학습시에는 청크들의 등장 순서에 shuffle


# tokenizer 생성, dataset 생성, dataloader 생성
def create_dataloader_v1(txt, batch_size=4, max_length=256,  stride=128, shuffle=True, drop_last=True, num_workers=0):
    
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # 모델 학습시에는 청크들의 등장 순서에 shuffle=True
        drop_last=drop_last,
        num_workers=num_workers  # 데이터 로딩 속도를 높이기 위해 사용하는 서브 프로세스(일꾼)의 수
    )

    return dataloader
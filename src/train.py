# src/train.py
from src.tokenizer import get_tokenizer
from src.data import create_dataloader_v1

def main():
    # with open("data/input.txt", "r", encoding="utf-8") as f:
    #     txt = f.read()
    txt = "hello, world!"

    tokenizer = get_tokenizer("gpt2")
    dataloader = create_dataloader_v1(txt, tokenizer)

if __name__ == "__main__":
    main()
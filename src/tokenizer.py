# src/tokenizer.py
import tiktoken

def get_tokenizer(name: str = "gpt2"):
    return tiktoken.get_encoding(name)

# tokenizer = tiktoken.get_encoding("gpt2")
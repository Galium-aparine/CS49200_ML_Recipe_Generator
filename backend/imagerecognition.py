# -*- coding: utf-8 -*-
"""
Recipe Generator

Original base: Hugging Face flax-community/t5-recipe-generation
Updated: Converted to PyTorch for faster generation
Original file: https://colab.research.google.com/drive/1srvHIRL7tIN1nIWq7hGBC-uYFyEjtDsa
"""

# === Install dependencies if needed ===
# !pip install transformers

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# === Model and Tokenizer Setup ===
MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    torch_dtype=torch.float16  # Use half precision for faster performance
).to("cuda").eval()

# === Generation Settings ===
prefix = "items: "
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95,
}

# === Special Tokens Mapping ===
special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n",
}

# === Helper Functions ===
def remove_special_tokens(text: str, special_tokens: list) -> str:
    """Remove all special tokens from a given text."""
    for token in special_tokens:
        text = text.replace(token, "")
    return text

def postprocess_generated_texts(texts, special_tokens) -> list:
    """Post-process model outputs by removing and replacing special tokens."""
    if not isinstance(texts, list):
        texts = [texts]

    processed_texts = []
    for text in texts:
        text = remove_special_tokens(text, special_tokens)
        for token, replacement in tokens_map.items():
            text = text.replace(token, replacement)
        processed_texts.append(text)

    return processed_texts

def generate_recipe(texts) -> list:
    """Generate recipe text based on input ingredients or prompt."""
    if not isinstance(texts, list):
        texts = [texts]

    inputs = [prefix + text for text in texts]

    encodings = tokenizer(
        inputs,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask,
            **generation_kwargs
        )

    decoded_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    return postprocess_generated_texts(decoded_texts, special_tokens)

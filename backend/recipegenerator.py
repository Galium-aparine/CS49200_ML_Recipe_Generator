# -*- coding: utf-8 -*-
"""
Recipe Generator using Flax (Hugging Face)

Original base: Hugging Face flax-community/t5-recipe-generation
Updated file: https://colab.research.google.com/github/Galium-aparine/CS49200_ML_Recipe_Generator/blob/maddie/RecipeGenerator.ipynb
"""

# === Imports ===
from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer

# === Model Setup ===
MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

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
    """Remove special tokens from text."""
    for token in special_tokens:
        text = text.replace(token, "")
    return text

def postprocess_generated_texts(texts, special_tokens) -> list:
    """Postprocess generated outputs to clean formatting."""
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
    """Generate recipes from a list of input items."""
    if not isinstance(texts, list):
        texts = [texts]

    prompts = [prefix + item for item in texts]

    encodings = tokenizer(
        prompts,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="jax"
    )

    output_ids = model.generate(
        input_ids=encodings.input_ids,
        attention_mask=encodings.attention_mask,
        **generation_kwargs
    ).sequences

    decoded_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    return postprocess_generated_texts(decoded_texts, special_tokens)

def print_recipe_sections(recipe_text: str):
    """Nicely format and print the sections of a generated recipe."""
    sections = recipe_text.split("\n")
    current_section = ""

    for section in sections:
        section = section.strip()
        if section.startswith("title:"):
            print(f"[TITLE]: {section.replace('title:', '').strip().capitalize()}")
            current_section = "TITLE"
        elif section.startswith("ingredients:"):
            ingredients = section.replace("ingredients:", "").split("--")
            print("[INGREDIENTS]:")
            print("\n".join(f"  - {i+1}: {item.strip().capitalize()}" for i, item in enumerate(ingredients)))
            current_section = "INGREDIENTS"
        elif section.startswith("directions:"):
            directions = section.replace("directions:", "").split("--")
            print("[DIRECTIONS]:")
            print("\n".join(f"  - {i+1}: {step.strip().capitalize()}" for i, step in enumerate(directions)))
            current_section = "DIRECTIONS"

    print("-" * 130)

# === Example Usage ===
if __name__ == "__main__":
    items = [
        "macaroni, butter, salt, bacon, milk, flour, pepper, cream corn",
        "provolone cheese, bacon, bread, ginger",
    ]

    generated_recipes = generate_recipe(items)

    for recipe in generated_recipes:
        print_recipe_sections(recipe)

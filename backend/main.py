# -*- coding: utf-8 -*-
"""
FastAPI Recipe Generation App

- Generate recipes using a fine-tuned Hugging Face model (PyTorch).
- Detect ingredients in images.
- Fetch related recipes from Spoonacular API.
"""

# === Imports ===
import os
import shutil
import requests
import cv2

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from imagerecognition import detect_ingredients, substitute_objects

# === Model Setup ===
MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

# === API Key ===
SPOONACULAR_API_KEY = "YOUR_SPOONACULAR_API_KEY"

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Static Files Setup ===
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# === Special Tokens Mapping ===
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

# === Helper Functions ===
def remove_special_tokens(text: str, special_tokens: list) -> str:
    """Remove special tokens from text."""
    for token in special_tokens:
        text = text.replace(token, "")
    return text

def postprocess_generated_texts(texts, special_tokens) -> list:
    """Postprocess generated texts: remove and replace special tokens."""
    if not isinstance(texts, list):
        texts = [texts]

    processed = []
    for text in texts:
        text = remove_special_tokens(text, special_tokens)
        for k, v in tokens_map.items():
            text = text.replace(k, v)
        processed.append(text)

    return processed

def build_prompt(ingredients: list, cuisine: str = None, allergies: str = None, max_time: int = None) -> str:
    """Build input prompt for recipe generation."""
    prompt = "items: " + ", ".join(ingredients)
    if cuisine and cuisine.lower() != "any":
        prompt += f" | cuisine: {cuisine}"
    if allergies:
        prompt += f" | avoid: {allergies}"
    if max_time:
        prompt += f" | max_time: {max_time} mins"
    return prompt

def generate_recipe(prompt: str) -> str:
    """Generate a recipe from the prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        max_length=512,
        min_length=64,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=60,
        top_p=0.95,
    )
    generated = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    final_output = postprocess_generated_texts(generated, tokenizer.all_special_tokens)
    return final_output[0]

def get_spoonacular_recipes(ingredients: list) -> list:
    """Fetch alternative recipes from Spoonacular API."""
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "ingredients": ",".join(ingredients),
        "number": 5,
        "apiKey": SPOONACULAR_API_KEY,
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else []

# === Request Models ===
class RecipeRequest(BaseModel):
    ingredients: str
    allergies: str = ""
    cuisine: str = "Any"
    max_time: int = 60

# === API Endpoints ===
@app.post("/generate")
def generate_recipe_api(req: RecipeRequest):
    """Endpoint to generate a recipe."""
    ingredients_list = [item.strip().lower() for item in req.ingredients.split(",") if item.strip()]
    allergies = req.allergies.strip().lower() if req.allergies else ""
    prompt = build_prompt(ingredients_list, req.cuisine, allergies, req.max_time)

    # Generate recipe output
    generated_text = generate_recipe(prompt)

    # Parse generated sections (optional: this can be made better)
    for i in range(5):
        sections = generated_text.split("\n")
        for section in sections:
            section = section.strip()
            if section.startswith("title:"):
                print(f"[TITLE]: {section.replace('title:', '').strip().capitalize()}")
            elif section.startswith("ingredients:"):
                ingredients = section.replace("ingredients:", "").split("--")
                print("[INGREDIENTS]:")
                print("\n".join(f"  - {i+1}: {item.strip().capitalize()}" for i, item in enumerate(ingredients)))
            elif section.startswith("directions:"):
                directions = section.replace("directions:", "").split("--")
                print("[DIRECTIONS]:")
                print("\n".join(f"  - {i+1}: {step.strip().capitalize()}" for i, step in enumerate(directions)))

        print("-" * 130)

    # (Optional) Uncomment if you want Spoonacular fallback:
    # recipes = [] if generated_text.strip() else get_spoonacular_recipes(ingredients_list)
    # return {"ai_recipe": generated_text, "spoonacular_recipes": recipes}

@app.post("/detect-ingredients")
async def detect_ingredients_api(file: UploadFile = File(...)):
    """Endpoint to detect ingredients from an uploaded image."""
    temp_path = os.path.join(STATIC_DIR, f"output_{file.filename}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[INFO] Image successfully saved at: {temp_path}")

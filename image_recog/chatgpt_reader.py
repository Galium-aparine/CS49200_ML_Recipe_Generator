# -*- coding: utf-8 -*-
"""
Ingredient Detection from Image using OpenAI GPT-4 Turbo Vision

- Reads an image
- Sends it to GPT-4V to extract food ingredients
- Expects API key as environment variable 'OPENAI_API_KEY'
"""

# === Imports ===
import os
import base64
from openai import OpenAI

# === OpenAI Client Setup ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Helper Function ===
def analyze_image(image_path: str) -> str:
    """
    Analyze an image and extract a list of food ingredients using GPT-4 Vision.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Detected ingredients formatted as a Python list string.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Encode image to base64
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # Send image and prompt to GPT-4V
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "List all food ingredients in this image."
                            " Be concise and make the output a list that can be input into another Python function."
                            " Only output the list between square brackets."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content

# === Example Usage ===
if __name__ == "__main__":
    try:
        ingredients = analyze_image("image_recog/fridge.jpg")
        print("Detected ingredients:", ingredients)
    except Exception as e:
        print(f"Error: {e}")

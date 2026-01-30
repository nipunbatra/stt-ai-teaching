#!/usr/bin/env python3
"""
Generate images using Gemini 2.0 Flash for lecture slides.

Usage:
    python generate_gemini_image.py "Your prompt here" output_filename.png

Example:
    python generate_gemini_image.py "Educational diagram of neural network" nn_diagram.png

The image will be saved to slides/images/week05/
"""

import os
import sys
import base64
from io import BytesIO
from pathlib import Path
import subprocess

def get_api_key():
    """Get Gemini API key from environment or zsh_secrets."""
    if 'GEMINI_API_KEY' in os.environ:
        return os.environ['GEMINI_API_KEY']

    # Try to get from zsh_secrets
    result = subprocess.run(
        ['bash', '-c', 'source ~/.zsh_secrets 2>/dev/null && echo $GEMINI_API_KEY'],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        return result.stdout.strip()

    print("Error: GEMINI_API_KEY not found")
    print("Set it with: export GEMINI_API_KEY='your-key'")
    sys.exit(1)


def generate_image(prompt: str, output_filename: str):
    """Generate an image from a prompt and save it."""
    from google import genai
    from google.genai import types
    from PIL import Image

    api_key = get_api_key()
    os.environ['GEMINI_API_KEY'] = api_key

    # Output path
    output_dir = Path(__file__).parent.parent.parent / "slides" / "images" / "week05"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    print(f"Generating image...")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Output: {output_path}")
    print()

    client = genai.Client(api_key=api_key)

    # Use 1080p (Full HD) - good quality for slides without being excessive
    response = client.models.generate_content(
        model='models/gemini-3-pro-image-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                image_size="1080p"  # Full HD - 1920x1080, plenty for slides
            )
        )
    )

    # Extract image
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                if hasattr(part.inline_data, 'data') and part.inline_data.data:
                    image_data = part.inline_data.data
                    if not isinstance(image_data, bytes):
                        image_data = base64.b64decode(image_data)

                    img = Image.open(BytesIO(image_data))
                    img.save(output_path, format='PNG')

                    print(f"✓ Saved: {output_path}")
                    print(f"  Size: {img.size[0]}x{img.size[1]} pixels")
                    return str(output_path)

    print("Error: No image in response")
    if hasattr(response, 'text') and response.text:
        print(f"Text response: {response.text}")
    return None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    prompt = sys.argv[1]
    output_filename = sys.argv[2]

    if not output_filename.endswith('.png'):
        output_filename += '.png'

    generate_image(prompt, output_filename)


if __name__ == "__main__":
    main()

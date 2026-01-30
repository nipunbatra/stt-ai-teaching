"""
Interactive Data Augmentation Demo with Gradio
=============================================
Run: python augmentation_demo_app.py
Then open: http://localhost:7860

This demo lets students interactively explore image augmentations.
"""

import gradio as gr
import numpy as np
from PIL import Image
from scipy.ndimage import rotate, zoom, gaussian_filter
import io

# Try to import albumentations, fall back to manual transforms
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Note: albumentations not installed, using scipy transforms")


def apply_augmentation(
    image,
    horizontal_flip,
    vertical_flip,
    rotation_angle,
    brightness,
    contrast,
    noise_level,
    blur_sigma,
    cutout_size,
):
    """Apply selected augmentations to the image."""
    if image is None:
        return None

    # Convert to numpy array
    img = np.array(image).astype(float)

    # Horizontal flip
    if horizontal_flip:
        img = np.fliplr(img)

    # Vertical flip
    if vertical_flip:
        img = np.flipud(img)

    # Rotation
    if rotation_angle != 0:
        img = rotate(img, rotation_angle, reshape=False, mode='reflect')

    # Brightness
    if brightness != 1.0:
        img = img * brightness

    # Contrast
    if contrast != 1.0:
        mean = img.mean()
        img = (img - mean) * contrast + mean

    # Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape)
        img = img + noise

    # Gaussian blur
    if blur_sigma > 0:
        if len(img.shape) == 3:
            for c in range(img.shape[2]):
                img[:, :, c] = gaussian_filter(img[:, :, c], sigma=blur_sigma)
        else:
            img = gaussian_filter(img, sigma=blur_sigma)

    # Cutout
    if cutout_size > 0:
        h, w = img.shape[:2]
        y = np.random.randint(0, max(1, h - cutout_size))
        x = np.random.randint(0, max(1, w - cutout_size))
        img[y:y+cutout_size, x:x+cutout_size] = 0

    # Clip and convert back
    img = np.clip(img, 0, 255).astype(np.uint8)

    return Image.fromarray(img)


def batch_augment(image, num_augments=9):
    """Generate multiple random augmentations."""
    if image is None:
        return [None] * 9

    results = []
    np.random.seed(42)  # For reproducibility in demo

    for i in range(num_augments):
        np.random.seed(i * 100)  # Different seed for each

        aug_img = apply_augmentation(
            image,
            horizontal_flip=np.random.random() < 0.5,
            vertical_flip=False,  # Usually dangerous
            rotation_angle=np.random.uniform(-20, 20),
            brightness=np.random.uniform(0.7, 1.3),
            contrast=np.random.uniform(0.8, 1.2),
            noise_level=np.random.uniform(0, 15),
            blur_sigma=np.random.uniform(0, 1),
            cutout_size=int(np.random.uniform(0, 30)),
        )
        results.append(aug_img)

    return results


# Create the Gradio interface
with gr.Blocks(title="Data Augmentation Demo") as demo:
    gr.Markdown("""
    # 🎨 Interactive Data Augmentation Demo

    **Explore how different augmentations transform images!**

    Upload an image and adjust the sliders to see augmentations in real-time.
    """)

    with gr.Tab("Manual Controls"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")

                gr.Markdown("### Geometric Transforms")
                h_flip = gr.Checkbox(label="Horizontal Flip", value=False)
                v_flip = gr.Checkbox(label="Vertical Flip (⚠️ Often dangerous!)", value=False)
                rotation = gr.Slider(-45, 45, value=0, step=1, label="Rotation (degrees)")

                gr.Markdown("### Color Transforms")
                brightness = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="Brightness")
                contrast = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="Contrast")

                gr.Markdown("### Noise & Blur")
                noise = gr.Slider(0, 50, value=0, step=5, label="Gaussian Noise (σ)")
                blur = gr.Slider(0, 5, value=0, step=0.5, label="Blur (σ)")

                gr.Markdown("### Advanced")
                cutout = gr.Slider(0, 100, value=0, step=10, label="Cutout Size (pixels)")

                apply_btn = gr.Button("Apply Augmentation", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(type="pil", label="Augmented Image")

        apply_btn.click(
            apply_augmentation,
            inputs=[input_image, h_flip, v_flip, rotation, brightness, contrast, noise, blur, cutout],
            outputs=output_image
        )

    with gr.Tab("Batch Augmentation"):
        gr.Markdown("""
        ### Generate 9 Random Augmentations

        See how random augmentation pipelines create diverse training data!
        """)

        with gr.Row():
            batch_input = gr.Image(type="pil", label="Input Image")
            batch_btn = gr.Button("Generate 9 Augmentations", variant="primary")

        with gr.Row():
            batch_outputs = [gr.Image(type="pil", label=f"Aug {i+1}") for i in range(3)]
        with gr.Row():
            batch_outputs += [gr.Image(type="pil", label=f"Aug {i+4}") for i in range(3)]
        with gr.Row():
            batch_outputs += [gr.Image(type="pil", label=f"Aug {i+7}") for i in range(3)]

        batch_btn.click(
            batch_augment,
            inputs=batch_input,
            outputs=batch_outputs
        )

    with gr.Tab("Dangerous Augmentations"):
        gr.Markdown("""
        ### ⚠️ When Augmentation Goes Wrong

        Some augmentations can **change the label**! Test it yourself.

        **Example**: Upload a digit "6" and apply vertical flip → becomes "9"!
        """)

        with gr.Row():
            danger_input = gr.Image(type="pil", label="Upload a '6' digit")
            with gr.Column():
                danger_vflip = gr.Checkbox(label="Vertical Flip", value=True)
                danger_btn = gr.Button("Apply (See the danger!)", variant="stop")
            danger_output = gr.Image(type="pil", label="Result (Is it still '6'?)")

        danger_btn.click(
            lambda img, vf: apply_augmentation(img, False, vf, 0, 1.0, 1.0, 0, 0, 0),
            inputs=[danger_input, danger_vflip],
            outputs=danger_output
        )

    gr.Markdown("""
    ---
    ### 📚 Key Takeaways

    1. **Always preserve the label** - if augmentation changes what the image represents, don't use it!
    2. **Start mild, increase gradually** - begin with small rotation angles, low noise, etc.
    3. **Domain matters** - medical images need different augmentations than natural photos
    4. **Test your pipeline** - generate augmented samples and verify you can still label them correctly

    ---
    *CS 203: Software Tools and Techniques for AI - Week 5*
    """)


if __name__ == "__main__":
    demo.launch(share=False)

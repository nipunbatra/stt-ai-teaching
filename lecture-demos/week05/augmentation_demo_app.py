"""
Interactive Data Augmentation Demo
==================================
Run: python augmentation_demo_app.py

Opens at http://localhost:7860 with sample images ready to use.
"""

import gradio as gr
import numpy as np
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter


def get_sample_images():
    """Load sample images for immediate use."""
    samples = []

    # Try to load cat from skimage
    try:
        from skimage import data as skdata
        cat = skdata.chelsea()
        cat_img = Image.fromarray(cat).resize((300, 300))
        samples.append(("Cat (skimage)", cat_img))
    except:
        pass

    # Try to load MNIST digit
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        # Get a "6" digit
        idx = np.where(mnist.target.astype(int) == 6)[0][0]
        digit = mnist.data[idx].reshape(28, 28)
        digit_img = Image.fromarray((digit).astype(np.uint8)).resize((150, 150))
        digit_rgb = Image.merge('RGB', [digit_img, digit_img, digit_img])
        samples.append(("Digit 6 (MNIST)", digit_rgb))
    except:
        pass

    # Create synthetic colored squares as fallback
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[20:80, 20:80] = [255, 100, 100]  # Red square
    img[70:150, 100:180] = [100, 100, 255]  # Blue square
    img[120:180, 30:90] = [100, 255, 100]  # Green square
    samples.append(("Colored Shapes", Image.fromarray(img)))

    return samples


def apply_augmentation(image, h_flip, v_flip, rotation, brightness, contrast, noise, blur, cutout):
    """Apply augmentations to image."""
    if image is None:
        return None

    img = np.array(image).astype(float)

    if h_flip:
        img = np.fliplr(img)
    if v_flip:
        img = np.flipud(img)
    if rotation != 0:
        img = rotate(img, rotation, reshape=False, mode='reflect')
    if brightness != 1.0:
        img = img * brightness
    if contrast != 1.0:
        mean = img.mean()
        img = (img - mean) * contrast + mean
    if noise > 0:
        img = img + np.random.normal(0, noise, img.shape)
    if blur > 0:
        for c in range(img.shape[2]) if len(img.shape) == 3 else [0]:
            if len(img.shape) == 3:
                img[:, :, c] = gaussian_filter(img[:, :, c], sigma=blur)
            else:
                img = gaussian_filter(img, sigma=blur)
    if cutout > 0:
        h, w = img.shape[:2]
        y, x = np.random.randint(0, max(1, h - cutout)), np.random.randint(0, max(1, w - cutout))
        img[y:y+cutout, x:x+cutout] = 0

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def batch_augment(image):
    """Generate 9 random augmentations."""
    if image is None:
        return [None] * 9

    results = []
    for i in range(9):
        np.random.seed(i * 100)
        aug = apply_augmentation(
            image,
            h_flip=np.random.random() < 0.5,
            v_flip=False,
            rotation=np.random.uniform(-20, 20),
            brightness=np.random.uniform(0.7, 1.3),
            contrast=np.random.uniform(0.8, 1.2),
            noise=np.random.uniform(0, 15),
            blur=np.random.uniform(0, 1),
            cutout=int(np.random.uniform(0, 40)),
        )
        results.append(aug)
    return results


# Load sample images
print("Loading sample images...")
SAMPLES = get_sample_images()
print(f"Loaded {len(SAMPLES)} sample images")

# Build interface
with gr.Blocks(title="Data Augmentation Demo") as demo:
    gr.Markdown("# 🎨 Data Augmentation Demo\nExplore how augmentations transform images. Select a sample or upload your own!")

    with gr.Tabs():
        # Tab 1: Interactive Controls
        with gr.TabItem("🎛️ Interactive"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Sample selector
                    sample_dropdown = gr.Dropdown(
                        choices=[s[0] for s in SAMPLES],
                        value=SAMPLES[0][0] if SAMPLES else None,
                        label="📷 Select Sample Image"
                    )
                    input_img = gr.Image(type="pil", label="Input", value=SAMPLES[0][1] if SAMPLES else None)

                with gr.Column(scale=1):
                    output_img = gr.Image(type="pil", label="Output")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Geometric**")
                    h_flip = gr.Checkbox(label="Horizontal Flip")
                    v_flip = gr.Checkbox(label="⚠️ Vertical Flip")
                    rotation = gr.Slider(-45, 45, 0, step=5, label="Rotation (°)")
                with gr.Column():
                    gr.Markdown("**Color**")
                    brightness = gr.Slider(0.3, 2.0, 1.0, step=0.1, label="Brightness")
                    contrast = gr.Slider(0.3, 2.0, 1.0, step=0.1, label="Contrast")
                with gr.Column():
                    gr.Markdown("**Effects**")
                    noise = gr.Slider(0, 50, 0, step=5, label="Noise (σ)")
                    blur = gr.Slider(0, 5, 0, step=0.5, label="Blur (σ)")
                    cutout = gr.Slider(0, 100, 0, step=10, label="Cutout (px)")

            apply_btn = gr.Button("Apply Augmentation", variant="primary", size="lg")

            # Load sample on dropdown change
            def load_sample(name):
                for n, img in SAMPLES:
                    if n == name:
                        return img
                return None

            sample_dropdown.change(load_sample, sample_dropdown, input_img)

            # Apply augmentation
            apply_btn.click(
                apply_augmentation,
                [input_img, h_flip, v_flip, rotation, brightness, contrast, noise, blur, cutout],
                output_img
            )

        # Tab 2: Batch Generation
        with gr.TabItem("🎲 Batch (1→9)"):
            gr.Markdown("Generate **9 random augmentations** from one image!")

            with gr.Row():
                batch_dropdown = gr.Dropdown(
                    choices=[s[0] for s in SAMPLES],
                    value=SAMPLES[0][0] if SAMPLES else None,
                    label="Select Sample"
                )
                batch_input = gr.Image(type="pil", label="Input", value=SAMPLES[0][1] if SAMPLES else None)

            batch_btn = gr.Button("Generate 9 Augmentations", variant="primary", size="lg")

            with gr.Row():
                batch_out = [gr.Image(type="pil", label=f"#{i+1}", show_label=True) for i in range(3)]
            with gr.Row():
                batch_out += [gr.Image(type="pil", label=f"#{i+4}", show_label=True) for i in range(3)]
            with gr.Row():
                batch_out += [gr.Image(type="pil", label=f"#{i+7}", show_label=True) for i in range(3)]

            batch_dropdown.change(load_sample, batch_dropdown, batch_input)
            batch_btn.click(batch_augment, batch_input, batch_out)

        # Tab 3: Dangerous Examples
        with gr.TabItem("⚠️ Dangerous"):
            gr.Markdown("""
            ### When Augmentation Breaks Labels

            Some augmentations can **change the meaning** of an image!

            **Try it**: Select the digit "6" and apply vertical flip → it becomes "9"!
            """)

            with gr.Row():
                danger_dropdown = gr.Dropdown(
                    choices=[s[0] for s in SAMPLES],
                    value="Digit 6 (MNIST)" if any("Digit" in s[0] for s in SAMPLES) else (SAMPLES[0][0] if SAMPLES else None),
                    label="Select Image"
                )
                danger_input = gr.Image(type="pil", label="Before")
                danger_output = gr.Image(type="pil", label="After")

            with gr.Row():
                danger_vflip = gr.Checkbox(label="Vertical Flip", value=True)
                danger_rotate = gr.Checkbox(label="180° Rotation", value=False)

            danger_btn = gr.Button("Apply Dangerous Transform", variant="stop", size="lg")

            def danger_transform(img, vf, rot):
                if img is None:
                    return None
                return apply_augmentation(img, False, vf, 180 if rot else 0, 1.0, 1.0, 0, 0, 0)

            # Load digit by default for danger tab
            def load_danger_sample(name):
                for n, img in SAMPLES:
                    if n == name:
                        return img
                return None

            danger_dropdown.change(load_danger_sample, danger_dropdown, danger_input)
            danger_btn.click(danger_transform, [danger_input, danger_vflip, danger_rotate], danger_output)

            # Initialize with digit
            demo.load(lambda: load_danger_sample("Digit 6 (MNIST)") if any("Digit" in s[0] for s in SAMPLES) else (SAMPLES[0][1] if SAMPLES else None), outputs=danger_input)

    gr.Markdown("""
    ---
    **Key Rules**: (1) Preserve the label (2) Start mild (3) Test your pipeline

    *CS 203: Software Tools and Techniques for AI - Week 5*
    """)


if __name__ == "__main__":
    demo.launch()

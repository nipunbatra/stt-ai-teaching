"""
Interactive Data Augmentation Demo with Streamlit
=================================================
Run: streamlit run augmentation_demo_streamlit.py

This demo lets students interactively explore image augmentations.
"""

import streamlit as st
import numpy as np
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter
import io

st.set_page_config(
    page_title="Data Augmentation Demo",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Interactive Data Augmentation Demo")
st.markdown("**Explore how different augmentations transform images!**")


def apply_augmentation(img, h_flip, v_flip, rotation, brightness, contrast, noise, blur, cutout):
    """Apply augmentations to image."""
    img = img.astype(float)

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
        if len(img.shape) == 3:
            for c in range(img.shape[2]):
                img[:, :, c] = gaussian_filter(img[:, :, c], sigma=blur)
        else:
            img = gaussian_filter(img, sigma=blur)
    if cutout > 0:
        h, w = img.shape[:2]
        y = np.random.randint(0, max(1, h - cutout))
        x = np.random.randint(0, max(1, w - cutout))
        img[y:y+cutout, x:x+cutout] = 0

    return np.clip(img, 0, 255).astype(np.uint8)


# Sidebar controls
st.sidebar.header("🎛️ Augmentation Controls")

st.sidebar.subheader("Geometric")
h_flip = st.sidebar.checkbox("Horizontal Flip", value=False)
v_flip = st.sidebar.checkbox("⚠️ Vertical Flip (Dangerous!)", value=False)
rotation = st.sidebar.slider("Rotation (°)", -45, 45, 0)

st.sidebar.subheader("Color")
brightness = st.sidebar.slider("Brightness", 0.3, 2.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contrast", 0.3, 2.0, 1.0, 0.1)

st.sidebar.subheader("Noise & Blur")
noise = st.sidebar.slider("Gaussian Noise (σ)", 0, 50, 0, 5)
blur = st.sidebar.slider("Blur (σ)", 0.0, 5.0, 0.0, 0.5)

st.sidebar.subheader("Advanced")
cutout = st.sidebar.slider("Cutout Size (px)", 0, 100, 0, 10)

# Main area
tab1, tab2, tab3 = st.tabs(["📷 Single Image", "🎲 Batch Augmentation", "⚠️ Dangerous Examples"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            st.image(image, caption="Original", use_container_width=True)

    with col2:
        st.subheader("Augmented Image")
        if uploaded_file is not None:
            augmented = apply_augmentation(
                img_array, h_flip, v_flip, rotation,
                brightness, contrast, noise, blur, cutout
            )
            st.image(augmented, caption="Augmented", use_container_width=True)

            # Show what was applied
            applied = []
            if h_flip:
                applied.append("H-Flip")
            if v_flip:
                applied.append("V-Flip ⚠️")
            if rotation != 0:
                applied.append(f"Rotate {rotation}°")
            if brightness != 1.0:
                applied.append(f"Brightness ×{brightness:.1f}")
            if contrast != 1.0:
                applied.append(f"Contrast ×{contrast:.1f}")
            if noise > 0:
                applied.append(f"Noise σ={noise}")
            if blur > 0:
                applied.append(f"Blur σ={blur:.1f}")
            if cutout > 0:
                applied.append(f"Cutout {cutout}px")

            if applied:
                st.info(f"**Applied**: {', '.join(applied)}")
            else:
                st.info("No augmentation applied")
        else:
            st.info("👆 Upload an image to see augmentation")

with tab2:
    st.subheader("🎲 Generate 9 Random Augmentations")
    st.markdown("See how random augmentation creates diverse training data!")

    if uploaded_file is not None:
        if st.button("Generate Batch", type="primary"):
            cols = st.columns(3)
            for i in range(9):
                np.random.seed(i * 42)
                aug = apply_augmentation(
                    img_array,
                    h_flip=np.random.random() < 0.5,
                    v_flip=False,
                    rotation=np.random.uniform(-20, 20),
                    brightness=np.random.uniform(0.7, 1.3),
                    contrast=np.random.uniform(0.8, 1.2),
                    noise=np.random.uniform(0, 15),
                    blur=np.random.uniform(0, 1),
                    cutout=int(np.random.uniform(0, 30))
                )
                cols[i % 3].image(aug, caption=f"Augmentation {i+1}", use_container_width=True)
    else:
        st.info("👆 Upload an image first")

with tab3:
    st.subheader("⚠️ When Augmentation Goes Wrong")

    st.markdown("""
    Some augmentations can **change the label**!

    **Example**: A digit "6" becomes "9" with vertical flip!
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Before (Label: 6)")
        # Create a simple "6" digit
        six = np.zeros((100, 80), dtype=np.uint8)
        from scipy.ndimage import gaussian_filter as gf
        # Draw a 6-like shape
        Y, X = np.ogrid[:100, :80]
        # Top circle
        six[((Y-30)**2 + (X-45)**2) < 400] = 255
        six[((Y-30)**2 + (X-45)**2) < 200] = 0
        # Bottom circle
        six[((Y-65)**2 + (X-45)**2) < 400] = 255
        six[((Y-65)**2 + (X-45)**2) < 200] = 0
        # Connector
        six[25:70, 25:35] = 255
        six = gf(six, sigma=1)
        st.image(six, caption="Original '6'", width=150)

    with col2:
        st.markdown("### After Vertical Flip (Label: ???)")
        nine = np.flipud(six)
        st.image(nine, caption="Now looks like '9'!", width=150)

    st.error("**Lesson**: Always verify that augmentation preserves the label!")

# Footer
st.markdown("---")
st.markdown("""
### 📚 Key Takeaways

1. **Always preserve the label** - if augmentation changes meaning, don't use it!
2. **Start mild, increase gradually** - begin with small values
3. **Domain matters** - medical images need different augmentations than photos
4. **Test your pipeline** - generate samples and verify labels are preserved

---
*CS 203: Software Tools and Techniques for AI - Week 5*
""")

"""
Generate HIGH-RESOLUTION augmentation example images for week05 slides.
Uses REAL MNIST (28x28) via sklearn.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates
from sklearn.datasets import fetch_openml

# Create output directory
os.makedirs('../../slides/images/week05', exist_ok=True)

# Load REAL MNIST
print("Loading MNIST from sklearn...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.reshape(-1, 28, 28)
y = mnist.target.astype(int)
print(f"Loaded {len(X)} images")

def get_digit(label, idx=0):
    """Get a specific digit from MNIST."""
    indices = np.where(y == label)[0]
    img = X[indices[idx]].astype(float)
    return img

# =============================================================================
# 1. MNIST Digit Augmentation Examples (REAL MNIST)
# =============================================================================

digit_img = get_digit(3, idx=5)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

# Original
axes[0, 0].imshow(digit_img, cmap='gray', interpolation='nearest')
axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# Rotated 15 degrees
rotated = rotate(digit_img, 15, reshape=False, mode='constant', cval=0)
axes[0, 1].imshow(rotated, cmap='gray', interpolation='nearest')
axes[0, 1].set_title('Rotate +15°', fontsize=13)
axes[0, 1].axis('off')

# Rotated -15 degrees
rotated_neg = rotate(digit_img, -15, reshape=False, mode='constant', cval=0)
axes[0, 2].imshow(rotated_neg, cmap='gray', interpolation='nearest')
axes[0, 2].set_title('Rotate -15°', fontsize=13)
axes[0, 2].axis('off')

# Horizontal flip
flipped_h = np.fliplr(digit_img)
axes[0, 3].imshow(flipped_h, cmap='gray', interpolation='nearest')
axes[0, 3].set_title('H-Flip', fontsize=13)
axes[0, 3].axis('off')

# Zoom/Scale
zoomed = zoom(digit_img, 1.3)
h, w = zoomed.shape
start_h, start_w = (h - 28) // 2, (w - 28) // 2
zoomed_crop = zoomed[start_h:start_h+28, start_w:start_w+28]
axes[0, 4].imshow(zoomed_crop, cmap='gray', interpolation='nearest')
axes[0, 4].set_title('Zoom 1.3x', fontsize=13)
axes[0, 4].axis('off')

# Shifted right
shifted = np.roll(digit_img, 3, axis=1)
axes[1, 0].imshow(shifted, cmap='gray', interpolation='nearest')
axes[1, 0].set_title('Shift Right', fontsize=13)
axes[1, 0].axis('off')

# Add noise
np.random.seed(42)
noisy = digit_img + np.random.normal(0, 30, digit_img.shape)
noisy = np.clip(noisy, 0, 255)
axes[1, 1].imshow(noisy, cmap='gray', interpolation='nearest')
axes[1, 1].set_title('+ Noise', fontsize=13)
axes[1, 1].axis('off')

# Brighter
brighter = np.clip(digit_img * 1.5, 0, 255)
axes[1, 2].imshow(brighter, cmap='gray', interpolation='nearest')
axes[1, 2].set_title('Brighter', fontsize=13)
axes[1, 2].axis('off')

# Darker
darker = digit_img * 0.6
axes[1, 3].imshow(darker, cmap='gray', interpolation='nearest')
axes[1, 3].set_title('Darker', fontsize=13)
axes[1, 3].axis('off')

# Elastic deformation
def elastic_transform(image, alpha=3, sigma=0.5):
    random_state = np.random.RandomState(42)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y_grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y_grid + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)

elastic = elastic_transform(digit_img)
axes[1, 4].imshow(elastic, cmap='gray', interpolation='nearest')
axes[1, 4].set_title('Elastic', fontsize=13)
axes[1, 4].axis('off')

plt.suptitle('Image Augmentation: Same Digit "3" → 10 Training Examples!', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/digit_augmentation_examples.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: digit_augmentation_examples.png")

# =============================================================================
# 2. The "6 vs 9" Problem - BAD Augmentation (REAL MNIST)
# =============================================================================

digit_6 = get_digit(6, idx=0)
digit_9 = get_digit(9, idx=0)

fig, axes = plt.subplots(1, 4, figsize=(12, 4))

axes[0].imshow(digit_6, cmap='gray', interpolation='nearest')
axes[0].set_title('Original: "6"', fontsize=16, fontweight='bold')
axes[0].axis('off')

# Flip vertically
flipped_6 = np.flipud(digit_6)
axes[1].imshow(flipped_6, cmap='gray', interpolation='nearest')
axes[1].set_title('V-Flip: looks like "9"!', fontsize=16, color='red', fontweight='bold')
axes[1].axis('off')

axes[2].text(0.5, 0.5, '≠', fontsize=60, ha='center', va='center', transform=axes[2].transAxes)
axes[2].axis('off')

axes[3].imshow(digit_9, cmap='gray', interpolation='nearest')
axes[3].set_title('Actual "9"', fontsize=16, fontweight='bold')
axes[3].axis('off')

plt.suptitle('BAD Augmentation: Vertical Flip Changes the Label!', fontsize=18, fontweight='bold', color='red')
plt.tight_layout()
plt.savefig('../../slides/images/week05/bad_augmentation_6_vs_9.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: bad_augmentation_6_vs_9.png")

# =============================================================================
# 3. Good vs Bad Augmentation Table Visual (REAL MNIST)
# =============================================================================

digit_5 = get_digit(5, idx=0)
digit_6 = get_digit(6, idx=0)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Row 1: GOOD augmentations
axes[0, 0].imshow(digit_5, cmap='gray', interpolation='nearest')
axes[0, 0].set_title('Original "5"', fontsize=13)
axes[0, 0].axis('off')

axes[0, 1].imshow(np.fliplr(digit_5), cmap='gray', interpolation='nearest')
axes[0, 1].set_title('H-Flip: Still "5" ✓', fontsize=13, color='green')
axes[0, 1].axis('off')

axes[0, 2].imshow(rotate(digit_5, 12, reshape=False, mode='constant'), cmap='gray', interpolation='nearest')
axes[0, 2].set_title('Rotate: Still "5" ✓', fontsize=13, color='green')
axes[0, 2].axis('off')

axes[0, 3].imshow(digit_5 * 0.6, cmap='gray', interpolation='nearest')
axes[0, 3].set_title('Darker: Still "5" ✓', fontsize=13, color='green')
axes[0, 3].axis('off')

# Row 2: BAD augmentations
axes[1, 0].imshow(digit_6, cmap='gray', interpolation='nearest')
axes[1, 0].set_title('Original "6"', fontsize=13)
axes[1, 0].axis('off')

axes[1, 1].imshow(np.flipud(digit_6), cmap='gray', interpolation='nearest')
axes[1, 1].set_title('V-Flip: Now "9"! ✗', fontsize=13, color='red')
axes[1, 1].axis('off')

axes[1, 2].imshow(rotate(digit_6, 180, reshape=False, mode='constant'), cmap='gray', interpolation='nearest')
axes[1, 2].set_title('180° Rot: Now "9"! ✗', fontsize=13, color='red')
axes[1, 2].axis('off')

# Heavy blur
blurred = gaussian_filter(digit_5, sigma=3)
axes[1, 3].imshow(blurred, cmap='gray', interpolation='nearest')
axes[1, 3].set_title('Heavy Blur: ??? ✗', fontsize=13, color='red')
axes[1, 3].axis('off')

# Add row labels
fig.text(0.02, 0.72, 'GOOD', fontsize=16, fontweight='bold', color='green', rotation=90, va='center')
fig.text(0.02, 0.28, 'BAD', fontsize=16, fontweight='bold', color='red', rotation=90, va='center')

plt.suptitle('Augmentation Must Preserve the Label!', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(left=0.08)
plt.savefig('../../slides/images/week05/good_vs_bad_augmentation.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: good_vs_bad_augmentation.png")

# =============================================================================
# 4. Overfitting visualization
# =============================================================================

np.random.seed(42)
epochs = np.arange(1, 51)

train_acc_no_aug = 0.5 + 0.5 * (1 - np.exp(-epochs / 10))
test_acc_no_aug = 0.5 + 0.3 * (1 - np.exp(-epochs / 15)) - 0.1 * (1 - np.exp(-(epochs - 20) / 10)) * (epochs > 20)

train_acc_aug = 0.5 + 0.4 * (1 - np.exp(-epochs / 15))
test_acc_aug = 0.5 + 0.35 * (1 - np.exp(-epochs / 12))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, train_acc_no_aug, 'b-', linewidth=3, label='Train')
axes[0].plot(epochs, test_acc_no_aug, 'r-', linewidth=3, label='Test')
axes[0].fill_between(epochs, train_acc_no_aug, test_acc_no_aug, alpha=0.3, color='red', label='Overfitting Gap')
axes[0].set_xlabel('Epoch', fontsize=14)
axes[0].set_ylabel('Accuracy', fontsize=14)
axes[0].set_title('Without Augmentation', fontsize=16, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=12)
axes[0].set_ylim(0.4, 1.0)
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, train_acc_aug, 'b-', linewidth=3, label='Train')
axes[1].plot(epochs, test_acc_aug, 'g-', linewidth=3, label='Test')
axes[1].fill_between(epochs, train_acc_aug, test_acc_aug, alpha=0.3, color='green', label='Small Gap')
axes[1].set_xlabel('Epoch', fontsize=14)
axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_title('With Augmentation', fontsize=16, fontweight='bold')
axes[1].legend(loc='lower right', fontsize=12)
axes[1].set_ylim(0.4, 1.0)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Augmentation Reduces Overfitting', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/augmentation_overfitting.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: augmentation_overfitting.png")

# =============================================================================
# 5. Geometric Transforms Visual (REAL MNIST)
# =============================================================================

digit = get_digit(3, idx=2)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Rotation
axes[0, 0].imshow(digit, cmap='gray', interpolation='nearest')
axes[0, 0].set_title('Original', fontsize=14)
axes[0, 0].axis('off')

rotated = rotate(digit, 20, reshape=False, mode='constant')
axes[0, 1].imshow(rotated, cmap='gray', interpolation='nearest')
axes[0, 1].set_title('Rotation (+20°)', fontsize=14)
axes[0, 1].axis('off')

# Flip
flipped = np.fliplr(digit)
axes[0, 2].imshow(flipped, cmap='gray', interpolation='nearest')
axes[0, 2].set_title('Horizontal Flip', fontsize=14)
axes[0, 2].axis('off')

# Translation
translated = np.roll(np.roll(digit, 4, axis=1), -2, axis=0)
axes[1, 0].imshow(translated, cmap='gray', interpolation='nearest')
axes[1, 0].set_title('Translation', fontsize=14)
axes[1, 0].axis('off')

# Scale/Zoom
zoomed = zoom(digit, 1.4)
h, w = zoomed.shape
start_h, start_w = (h - 28) // 2, (w - 28) // 2
zoomed_crop = zoomed[start_h:start_h+28, start_w:start_w+28]
axes[1, 1].imshow(zoomed_crop, cmap='gray', interpolation='nearest')
axes[1, 1].set_title('Scaling (1.4x)', fontsize=14)
axes[1, 1].axis('off')

# Random Crop (simulated)
crop = digit[2:26, 4:28]
crop_resized = zoom(crop, 28/24)[:28, :28]
axes[1, 2].imshow(crop_resized, cmap='gray', interpolation='nearest')
axes[1, 2].set_title('Random Crop', fontsize=14)
axes[1, 2].axis('off')

plt.suptitle('Geometric Transforms', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/geometric_transforms.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: geometric_transforms.png")

# =============================================================================
# 6. Color Transforms Visual (REAL MNIST)
# =============================================================================

digit = get_digit(7, idx=0)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(digit, cmap='gray', interpolation='nearest')
axes[0, 0].set_title('Original', fontsize=14)
axes[0, 0].axis('off')

# Brightness up
brighter = np.clip(digit * 1.6, 0, 255)
axes[0, 1].imshow(brighter, cmap='gray', interpolation='nearest')
axes[0, 1].set_title('Brighter (+60%)', fontsize=14)
axes[0, 1].axis('off')

# Brightness down
darker = digit * 0.5
axes[0, 2].imshow(darker, cmap='gray', interpolation='nearest')
axes[0, 2].set_title('Darker (-50%)', fontsize=14)
axes[0, 2].axis('off')

# Contrast up
mean_val = digit.mean()
high_contrast = (digit - mean_val) * 1.5 + mean_val
high_contrast = np.clip(high_contrast, 0, 255)
axes[1, 0].imshow(high_contrast, cmap='gray', interpolation='nearest')
axes[1, 0].set_title('High Contrast', fontsize=14)
axes[1, 0].axis('off')

# Contrast down
low_contrast = (digit - mean_val) * 0.5 + mean_val
axes[1, 1].imshow(low_contrast, cmap='gray', interpolation='nearest')
axes[1, 1].set_title('Low Contrast', fontsize=14)
axes[1, 1].axis('off')

# Invert
inverted = 255 - digit
axes[1, 2].imshow(inverted, cmap='gray', interpolation='nearest')
axes[1, 2].set_title('Inverted', fontsize=14)
axes[1, 2].axis('off')

plt.suptitle('Color/Intensity Transforms', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/color_transforms.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: color_transforms.png")

# =============================================================================
# 7. Mixup Visualization (REAL MNIST)
# =============================================================================

digit_3 = get_digit(3, idx=0)
digit_7 = get_digit(7, idx=0)

fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))

axes[0].imshow(digit_3, cmap='gray', interpolation='nearest')
axes[0].set_title('Image A: "3"\nLabel: [1, 0]', fontsize=13)
axes[0].axis('off')

axes[1].text(0.5, 0.5, '+', fontsize=40, ha='center', va='center', transform=axes[1].transAxes)
axes[1].axis('off')

axes[2].imshow(digit_7, cmap='gray', interpolation='nearest')
axes[2].set_title('Image B: "7"\nLabel: [0, 1]', fontsize=13)
axes[2].axis('off')

axes[3].text(0.5, 0.5, '=', fontsize=40, ha='center', va='center', transform=axes[3].transAxes)
axes[3].axis('off')

lam = 0.6
mixed = lam * digit_3 + (1 - lam) * digit_7
axes[4].imshow(mixed, cmap='gray', interpolation='nearest')
axes[4].set_title(f'Mixup (λ={lam})\nLabel: [{lam}, {1-lam}]', fontsize=13)
axes[4].axis('off')

plt.suptitle('Mixup: Blend Images AND Labels', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/mixup_example.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: mixup_example.png")

# =============================================================================
# 8. Cutout Visualization (REAL MNIST)
# =============================================================================

digit = get_digit(3, idx=3)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

axes[0].imshow(digit, cmap='gray', interpolation='nearest')
axes[0].set_title('Original', fontsize=14)
axes[0].axis('off')

# Cutout 1
cutout1 = digit.copy()
cutout1[5:13, 5:13] = 0
axes[1].imshow(cutout1, cmap='gray', interpolation='nearest')
axes[1].set_title('Cutout 1', fontsize=14)
axes[1].axis('off')

# Cutout 2
cutout2 = digit.copy()
cutout2[12:20, 14:22] = 0
axes[2].imshow(cutout2, cmap='gray', interpolation='nearest')
axes[2].set_title('Cutout 2', fontsize=14)
axes[2].axis('off')

# Cutout 3
cutout3 = digit.copy()
cutout3[3:11, 16:24] = 0
axes[3].imshow(cutout3, cmap='gray', interpolation='nearest')
axes[3].set_title('Cutout 3', fontsize=14)
axes[3].axis('off')

plt.suptitle('Cutout: Random Patch Removal (Still a "3"!)', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/cutout_example.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: cutout_example.png")

# =============================================================================
# 9. Text Augmentation Examples
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')

text_examples = [
    ("Original:", "This movie was absolutely fantastic!", "black", "bold"),
    ("Synonym:", "This film was absolutely fantastic!", "green", "normal"),
    ("Synonym:", "This movie was really fantastic!", "green", "normal"),
    ("Synonym:", "This movie was absolutely amazing!", "green", "normal"),
    ("Back-trans:", "This film was truly wonderful!", "blue", "normal"),
    ("Paraphrase:", "I thought this movie was great!", "purple", "normal"),
]

y_pos = 0.88
for label, text, color, weight in text_examples:
    ax.text(0.02, y_pos, label, fontsize=14, fontweight='bold', color='gray', transform=ax.transAxes)
    ax.text(0.18, y_pos, text, fontsize=16, fontweight=weight, color=color, transform=ax.transAxes)
    ax.text(0.82, y_pos, "POSITIVE", fontsize=13, color='green', fontweight='bold',
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    y_pos -= 0.13

ax.text(0.5, 0.08, "All 6 examples preserve the POSITIVE sentiment!",
        fontsize=15, ha='center', transform=ax.transAxes, style='italic')

plt.title('Text Augmentation: Same Sentiment, Different Words', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../../slides/images/week05/text_augmentation_examples.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: text_augmentation_examples.png")

# =============================================================================
# 10. Noise Augmentation (REAL MNIST)
# =============================================================================

digit = get_digit(5, idx=2)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

axes[0].imshow(digit, cmap='gray', interpolation='nearest')
axes[0].set_title('Original', fontsize=14)
axes[0].axis('off')

# Gaussian noise
np.random.seed(123)
noisy1 = digit + np.random.normal(0, 25, digit.shape)
noisy1 = np.clip(noisy1, 0, 255)
axes[1].imshow(noisy1, cmap='gray', interpolation='nearest')
axes[1].set_title('Gaussian Noise', fontsize=14)
axes[1].axis('off')

# Salt and pepper
np.random.seed(456)
noisy2 = digit.copy()
salt = np.random.random(digit.shape) < 0.03
pepper = np.random.random(digit.shape) < 0.03
noisy2[salt] = 255
noisy2[pepper] = 0
axes[2].imshow(noisy2, cmap='gray', interpolation='nearest')
axes[2].set_title('Salt & Pepper', fontsize=14)
axes[2].axis('off')

# Speckle noise
np.random.seed(789)
noisy3 = digit + digit * np.random.normal(0, 0.3, digit.shape)
noisy3 = np.clip(noisy3, 0, 255)
axes[3].imshow(noisy3, cmap='gray', interpolation='nearest')
axes[3].set_title('Speckle Noise', fontsize=14)
axes[3].axis('off')

plt.suptitle('Noise Augmentation: Training for Robustness', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/noise_augmentation.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: noise_augmentation.png")

# =============================================================================
# 11. Blur Augmentation (REAL MNIST)
# =============================================================================

digit = get_digit(8, idx=0)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

axes[0].imshow(digit, cmap='gray', interpolation='nearest')
axes[0].set_title('Original', fontsize=14)
axes[0].axis('off')

# Gaussian blur light
blur1 = gaussian_filter(digit, sigma=0.8)
axes[1].imshow(blur1, cmap='gray', interpolation='nearest')
axes[1].set_title('Light Blur (σ=0.8)', fontsize=14)
axes[1].axis('off')

# Medium blur
blur2 = gaussian_filter(digit, sigma=1.5)
axes[2].imshow(blur2, cmap='gray', interpolation='nearest')
axes[2].set_title('Medium Blur (σ=1.5)', fontsize=14)
axes[2].axis('off')

# Heavy blur - too much!
blur3 = gaussian_filter(digit, sigma=3)
axes[3].imshow(blur3, cmap='gray', interpolation='nearest')
axes[3].set_title('Heavy Blur (σ=3) ⚠️', fontsize=14, color='orange')
axes[3].axis('off')

plt.suptitle('Blur: Use Sparingly!', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/blur_augmentation.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: blur_augmentation.png")

# =============================================================================
# 12. Audio Spectrogram Augmentation (Simulated)
# =============================================================================

# Create a realistic-looking spectrogram
np.random.seed(42)
t = np.linspace(0, 2, 100)
f = np.linspace(0, 8000, 80)
T, F = np.meshgrid(t, f)
spec = np.sin(2 * np.pi * 0.5 * T) * np.exp(-((F - 2000)**2) / (2 * 500**2))
spec += np.sin(2 * np.pi * 1 * T) * np.exp(-((F - 4000)**2) / (2 * 300**2))
spec += 0.1 * np.random.randn(*spec.shape)
spec = np.clip(spec, 0, 1)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

axes[0].imshow(spec, aspect='auto', cmap='viridis', origin='lower')
axes[0].set_title('Original Spectrogram', fontsize=13)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_xlabel('Time', fontsize=11)

# Time masking
spec_time = spec.copy()
spec_time[:, 30:50] = 0
axes[1].imshow(spec_time, aspect='auto', cmap='viridis', origin='lower')
axes[1].set_title('Time Masking', fontsize=13)
axes[1].set_xlabel('Time', fontsize=11)

# Frequency masking
spec_freq = spec.copy()
spec_freq[25:45, :] = 0
axes[2].imshow(spec_freq, aspect='auto', cmap='viridis', origin='lower')
axes[2].set_title('Frequency Masking', fontsize=13)
axes[2].set_xlabel('Time', fontsize=11)

# Both
spec_both = spec.copy()
spec_both[:, 60:75] = 0
spec_both[50:65, :] = 0
axes[3].imshow(spec_both, aspect='auto', cmap='viridis', origin='lower')
axes[3].set_title('SpecAugment', fontsize=13)
axes[3].set_xlabel('Time', fontsize=11)

plt.suptitle('Audio Augmentation: SpecAugment', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/specaugment_example.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: specaugment_example.png")

# =============================================================================
# 13. RandAugment Visualization (REAL MNIST)
# =============================================================================

digit = get_digit(4, idx=0)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

axes[0, 0].imshow(digit, cmap='gray', interpolation='nearest')
axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Random augmentation combinations
np.random.seed(1)
for i in range(1, 10):
    ax = axes.flat[i]
    aug = digit.copy()

    # Random rotation
    if np.random.random() < 0.5:
        angle = np.random.uniform(-20, 20)
        aug = rotate(aug, angle, reshape=False, mode='constant')

    # Random brightness
    if np.random.random() < 0.5:
        factor = np.random.uniform(0.7, 1.3)
        aug = np.clip(aug * factor, 0, 255)

    # Random noise
    if np.random.random() < 0.3:
        aug = aug + np.random.normal(0, 15, aug.shape)
        aug = np.clip(aug, 0, 255)

    # Random flip
    if np.random.random() < 0.3:
        aug = np.fliplr(aug)

    ax.imshow(aug, cmap='gray', interpolation='nearest')
    ax.set_title(f'Augment {i}', fontsize=11)
    ax.axis('off')

plt.suptitle('RandAugment: Random Combinations Each Epoch', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/randaugment_example.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: randaugment_example.png")

# =============================================================================
# 14. Cat Augmentation Example (Real Photo from skimage)
# =============================================================================

from PIL import Image

try:
    from skimage import data as skdata
    print("Loading cat from skimage...")
    cat_np = skdata.chelsea()  # Real cat photo included in skimage
    cat_np = np.array(Image.fromarray(cat_np).resize((256, 256)))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    # Original
    axes[0, 0].imshow(cat_np)
    axes[0, 0].set_title('Original Cat', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    # Rotated
    from scipy.ndimage import rotate as nd_rotate
    rotated_cat = nd_rotate(cat_np, 15, reshape=False, mode='reflect')
    axes[0, 1].imshow(rotated_cat)
    axes[0, 1].set_title('Rotated +15°', fontsize=13)
    axes[0, 1].axis('off')

    # Horizontal flip
    flipped_cat = np.fliplr(cat_np)
    axes[0, 2].imshow(flipped_cat)
    axes[0, 2].set_title('Horizontal Flip', fontsize=13)
    axes[0, 2].axis('off')

    # Darker
    darker_cat = (cat_np * 0.6).astype(np.uint8)
    axes[0, 3].imshow(darker_cat)
    axes[0, 3].set_title('Darker (-40%)', fontsize=13)
    axes[0, 3].axis('off')

    # Brighter
    brighter_cat = np.clip(cat_np * 1.4, 0, 255).astype(np.uint8)
    axes[1, 0].imshow(brighter_cat)
    axes[1, 0].set_title('Brighter (+40%)', fontsize=13)
    axes[1, 0].axis('off')

    # Noise
    np.random.seed(42)
    noisy_cat = cat_np.astype(float) + np.random.normal(0, 20, cat_np.shape)
    noisy_cat = np.clip(noisy_cat, 0, 255).astype(np.uint8)
    axes[1, 1].imshow(noisy_cat)
    axes[1, 1].set_title('+ Gaussian Noise', fontsize=13)
    axes[1, 1].axis('off')

    # Zoom/Crop
    h, w = cat_np.shape[:2]
    crop_size = int(h * 0.7)
    start = (h - crop_size) // 2
    cropped = cat_np[start:start+crop_size, start:start+crop_size]
    cropped_resized = np.array(Image.fromarray(cropped).resize((256, 256)))
    axes[1, 2].imshow(cropped_resized)
    axes[1, 2].set_title('Center Crop + Resize', fontsize=13)
    axes[1, 2].axis('off')

    # Blur
    from scipy.ndimage import gaussian_filter as gf
    blurred_cat = np.stack([gf(cat_np[:,:,c], sigma=2) for c in range(3)], axis=2)
    axes[1, 3].imshow(blurred_cat.astype(np.uint8))
    axes[1, 3].set_title('Gaussian Blur', fontsize=13)
    axes[1, 3].axis('off')

    plt.suptitle('Image Augmentation on Real Photos: All Still a Cat!', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../../slides/images/week05/cat_augmentation_example.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: cat_augmentation_example.png")
except Exception as e:
    print(f"Could not download cat image: {e}")

print("\n✓ All images generated with REAL MNIST!")

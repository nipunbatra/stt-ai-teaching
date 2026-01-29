"""
Generate augmentation example images for week05 slides.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from PIL import Image, ImageEnhance, ImageFilter
import os

# Create output directory
os.makedirs('../../slides/images/week05', exist_ok=True)

# =============================================================================
# 1. MNIST Digit Augmentation Examples
# =============================================================================

digits = load_digits()
# Get a clear "3" digit
idx = np.where(digits.target == 3)[0][5]
digit_img = digits.images[idx]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Original
axes[0, 0].imshow(digit_img, cmap='gray')
axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Rotated 15 degrees
from scipy.ndimage import rotate
rotated = rotate(digit_img, 15, reshape=False, mode='constant', cval=0)
axes[0, 1].imshow(rotated, cmap='gray')
axes[0, 1].set_title('Rotate 15°', fontsize=12)
axes[0, 1].axis('off')

# Rotated -15 degrees
rotated_neg = rotate(digit_img, -15, reshape=False, mode='constant', cval=0)
axes[0, 2].imshow(rotated_neg, cmap='gray')
axes[0, 2].set_title('Rotate -15°', fontsize=12)
axes[0, 2].axis('off')

# Horizontal flip
flipped_h = np.fliplr(digit_img)
axes[0, 3].imshow(flipped_h, cmap='gray')
axes[0, 3].set_title('H-Flip', fontsize=12)
axes[0, 3].axis('off')

# Zoom/Scale
from scipy.ndimage import zoom
zoomed = zoom(digit_img, 1.2)[1:9, 1:9]  # Crop to original size
axes[0, 4].imshow(zoomed, cmap='gray')
axes[0, 4].set_title('Zoom 1.2x', fontsize=12)
axes[0, 4].axis('off')

# Shifted right
shifted = np.roll(digit_img, 1, axis=1)
axes[1, 0].imshow(shifted, cmap='gray')
axes[1, 0].set_title('Shift Right', fontsize=12)
axes[1, 0].axis('off')

# Add noise
noisy = digit_img + np.random.normal(0, 2, digit_img.shape)
noisy = np.clip(noisy, 0, 16)
axes[1, 1].imshow(noisy, cmap='gray')
axes[1, 1].set_title('+ Noise', fontsize=12)
axes[1, 1].axis('off')

# Brighter
brighter = np.clip(digit_img * 1.3, 0, 16)
axes[1, 2].imshow(brighter, cmap='gray')
axes[1, 2].set_title('Brighter', fontsize=12)
axes[1, 2].axis('off')

# Darker
darker = digit_img * 0.7
axes[1, 3].imshow(darker, cmap='gray')
axes[1, 3].set_title('Darker', fontsize=12)
axes[1, 3].axis('off')

# Elastic deformation (simplified)
from scipy.ndimage import map_coordinates, gaussian_filter
def elastic_transform(image, alpha=2, sigma=0.5):
    random_state = np.random.RandomState(42)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)

elastic = elastic_transform(digit_img)
axes[1, 4].imshow(elastic, cmap='gray')
axes[1, 4].set_title('Elastic', fontsize=12)
axes[1, 4].axis('off')

plt.suptitle('Image Augmentation: Same Digit "3", 10 Training Examples!', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/digit_augmentation_examples.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: digit_augmentation_examples.png")

# =============================================================================
# 2. The "6 vs 9" Problem - BAD Augmentation
# =============================================================================

# Get a "6" and show vertical flip makes it "9"
idx_6 = np.where(digits.target == 6)[0][0]
digit_6 = digits.images[idx_6]

idx_9 = np.where(digits.target == 9)[0][0]
digit_9 = digits.images[idx_9]

fig, axes = plt.subplots(1, 4, figsize=(10, 3))

axes[0].imshow(digit_6, cmap='gray')
axes[0].set_title('Original: "6"', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Flip vertically
flipped_6 = np.flipud(digit_6)
axes[1].imshow(flipped_6, cmap='gray')
axes[1].set_title('V-Flip: looks like "9"!', fontsize=14, color='red', fontweight='bold')
axes[1].axis('off')

axes[2].text(0.5, 0.5, '!=', fontsize=40, ha='center', va='center', transform=axes[2].transAxes)
axes[2].axis('off')

axes[3].imshow(digit_9, cmap='gray')
axes[3].set_title('Actual "9"', fontsize=14, fontweight='bold')
axes[3].axis('off')

plt.suptitle('BAD Augmentation: Vertical Flip Changes the Label!', fontsize=14, fontweight='bold', color='red')
plt.tight_layout()
plt.savefig('../../slides/images/week05/bad_augmentation_6_vs_9.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: bad_augmentation_6_vs_9.png")

# =============================================================================
# 3. Good vs Bad Augmentation Table Visual
# =============================================================================

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# Row 1: GOOD augmentations for image classification
digit = digits.images[5]  # A "5"

axes[0, 0].imshow(digit, cmap='gray')
axes[0, 0].set_title('Original "5"', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(np.fliplr(digit), cmap='gray')
axes[0, 1].set_title('H-Flip: Still "5"', fontsize=11, color='green')
axes[0, 1].axis('off')

axes[0, 2].imshow(rotate(digit, 10, reshape=False, mode='constant'), cmap='gray')
axes[0, 2].set_title('Rotate: Still "5"', fontsize=11, color='green')
axes[0, 2].axis('off')

axes[0, 3].imshow(digit * 0.7, cmap='gray')
axes[0, 3].set_title('Darker: Still "5"', fontsize=11, color='green')
axes[0, 3].axis('off')

# Row 2: BAD augmentations
axes[1, 0].imshow(digit_6, cmap='gray')
axes[1, 0].set_title('Original "6"', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(np.flipud(digit_6), cmap='gray')
axes[1, 1].set_title('V-Flip: Now "9"!', fontsize=11, color='red')
axes[1, 1].axis('off')

axes[1, 2].imshow(rotate(digit_6, 180, reshape=False, mode='constant'), cmap='gray')
axes[1, 2].set_title('180° Rot: Now "9"!', fontsize=11, color='red')
axes[1, 2].axis('off')

# Heavy blur loses info
from scipy.ndimage import gaussian_filter
blurred = gaussian_filter(digit, sigma=2)
axes[1, 3].imshow(blurred, cmap='gray')
axes[1, 3].set_title('Heavy Blur: ???', fontsize=11, color='red')
axes[1, 3].axis('off')

# Add row labels
fig.text(0.02, 0.75, 'GOOD', fontsize=14, fontweight='bold', color='green', rotation=90, va='center')
fig.text(0.02, 0.25, 'BAD', fontsize=14, fontweight='bold', color='red', rotation=90, va='center')

plt.suptitle('Augmentation Must Preserve the Label!', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(left=0.08)
plt.savefig('../../slides/images/week05/good_vs_bad_augmentation.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: good_vs_bad_augmentation.png")

# =============================================================================
# 4. Overfitting visualization - with/without augmentation
# =============================================================================

np.random.seed(42)
epochs = np.arange(1, 51)

# Without augmentation - overfits
train_acc_no_aug = 0.5 + 0.5 * (1 - np.exp(-epochs / 10))
test_acc_no_aug = 0.5 + 0.3 * (1 - np.exp(-epochs / 15)) - 0.1 * (1 - np.exp(-(epochs - 20) / 10)) * (epochs > 20)

# With augmentation - better generalization
train_acc_aug = 0.5 + 0.4 * (1 - np.exp(-epochs / 15))
test_acc_aug = 0.5 + 0.35 * (1 - np.exp(-epochs / 12))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Without augmentation
axes[0].plot(epochs, train_acc_no_aug, 'b-', linewidth=2, label='Train')
axes[0].plot(epochs, test_acc_no_aug, 'r-', linewidth=2, label='Test')
axes[0].fill_between(epochs, train_acc_no_aug, test_acc_no_aug, alpha=0.3, color='red', label='Overfitting Gap')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Without Augmentation', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].set_ylim(0.4, 1.0)
axes[0].grid(True, alpha=0.3)

# With augmentation
axes[1].plot(epochs, train_acc_aug, 'b-', linewidth=2, label='Train')
axes[1].plot(epochs, test_acc_aug, 'g-', linewidth=2, label='Test')
axes[1].fill_between(epochs, train_acc_aug, test_acc_aug, alpha=0.3, color='green', label='Small Gap')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('With Augmentation', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].set_ylim(0.4, 1.0)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Augmentation Reduces Overfitting', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/augmentation_overfitting.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: augmentation_overfitting.png")

# =============================================================================
# 5. Text Augmentation Examples
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

text_examples = [
    ("Original:", "This movie was absolutely fantastic!", "black", "bold"),
    ("Synonym:", "This film was absolutely fantastic!", "green", "normal"),
    ("Synonym:", "This movie was really fantastic!", "green", "normal"),
    ("Synonym:", "This movie was absolutely amazing!", "green", "normal"),
    ("Back-trans:", "This film was truly wonderful!", "blue", "normal"),
    ("Paraphrase:", "I thought this movie was great!", "purple", "normal"),
]

y_pos = 0.9
for label, text, color, weight in text_examples:
    ax.text(0.02, y_pos, label, fontsize=12, fontweight='bold', color='gray', transform=ax.transAxes)
    ax.text(0.18, y_pos, text, fontsize=14, fontweight=weight, color=color, transform=ax.transAxes)
    ax.text(0.85, y_pos, "POSITIVE", fontsize=11, color='green', fontweight='bold',
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    y_pos -= 0.12

ax.text(0.5, 0.05, "All 6 examples preserve the POSITIVE sentiment!",
        fontsize=13, ha='center', transform=ax.transAxes, style='italic')

plt.title('Text Augmentation: Same Sentiment, Different Words', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../../slides/images/week05/text_augmentation_examples.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: text_augmentation_examples.png")

# =============================================================================
# 6. Mixup Visualization
# =============================================================================

# Get a "3" and a "7"
idx_3 = np.where(digits.target == 3)[0][0]
idx_7 = np.where(digits.target == 7)[0][0]
digit_3 = digits.images[idx_3]
digit_7 = digits.images[idx_7]

fig, axes = plt.subplots(1, 5, figsize=(12, 3))

axes[0].imshow(digit_3, cmap='gray')
axes[0].set_title('Image A: "3"\nLabel: [1, 0]', fontsize=11)
axes[0].axis('off')

axes[1].text(0.5, 0.5, '+', fontsize=30, ha='center', va='center', transform=axes[1].transAxes)
axes[1].axis('off')

axes[2].imshow(digit_7, cmap='gray')
axes[2].set_title('Image B: "7"\nLabel: [0, 1]', fontsize=11)
axes[2].axis('off')

axes[3].text(0.5, 0.5, '=', fontsize=30, ha='center', va='center', transform=axes[3].transAxes)
axes[3].axis('off')

# Mixup with lambda=0.6
lam = 0.6
mixed = lam * digit_3 + (1 - lam) * digit_7
axes[4].imshow(mixed, cmap='gray')
axes[4].set_title(f'Mixup (λ={lam})\nLabel: [{lam}, {1-lam}]', fontsize=11)
axes[4].axis('off')

plt.suptitle('Mixup: Blend Images AND Labels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/mixup_example.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: mixup_example.png")

# =============================================================================
# 7. Cutout Visualization
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(10, 3))

digit = digits.images[idx_3].copy()
axes[0].imshow(digit, cmap='gray')
axes[0].set_title('Original', fontsize=12)
axes[0].axis('off')

# Cutout 1
cutout1 = digit.copy()
cutout1[2:5, 2:5] = 0
axes[1].imshow(cutout1, cmap='gray')
axes[1].set_title('Cutout 1', fontsize=12)
axes[1].axis('off')

# Cutout 2
cutout2 = digit.copy()
cutout2[4:7, 4:7] = 0
axes[2].imshow(cutout2, cmap='gray')
axes[2].set_title('Cutout 2', fontsize=12)
axes[2].axis('off')

# Cutout 3
cutout3 = digit.copy()
cutout3[1:4, 5:8] = 0
axes[3].imshow(cutout3, cmap='gray')
axes[3].set_title('Cutout 3', fontsize=12)
axes[3].axis('off')

plt.suptitle('Cutout: Random Patch Removal (Still a "3"!)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../../slides/images/week05/cutout_example.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: cutout_example.png")

print("\nAll images generated successfully!")

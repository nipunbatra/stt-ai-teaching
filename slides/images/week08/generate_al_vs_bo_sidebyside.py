"""Generate side-by-side AL vs BayesOpt comparison for each iteration (0-9).
Each image shows AL on left, BayesOpt on right at the same iteration."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

base = '/Users/nipun/git/stt-ai-teaching/slides/images/week08'

for i in range(10):
    al_path = os.path.join(base, f'distill_al_{i}.png')
    bo_path = os.path.join(base, f'distill_ei_{i}.png')

    al_img = mpimg.imread(al_path)
    bo_img = mpimg.imread(bo_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.imshow(al_img)
    ax1.set_title(f'Active Learning — Iteration {i}', fontsize=14, fontweight='bold', color='#9C27B0')
    ax1.axis('off')

    ax2.imshow(bo_img)
    ax2.set_title(f'Bayesian Optimization — Iteration {i}', fontsize=14, fontweight='bold', color='#4CAF50')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(base, f'al_vs_bo_{i}.png'), dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()

print("Generated al_vs_bo_0.png through al_vs_bo_9.png")

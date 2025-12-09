#!/usr/bin/env python3
"""
Generate Git Branch Creation Workflow showing three states:
1. Initial state with HEAD->main
2. After creating branch (both point to same commit)
3. After switching to feature branch
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os

OUTPUT_DIR = "../figures"
OUTPUT_FILE = "git_branch_creation_workflow.png"

COLORS = {
    'head': '#ffe1f5',
    'branch_main': '#e8f5e9',
    'branch_feature': '#fff4e1',
    'commit': '#e1f5ff',
}

def add_commit_node(ax, x, y, label, color='#e1f5ff', size=0.7):
    """Add a commit node (circle)."""
    circle = Circle((x, y), size, facecolor=color, edgecolor='black', linewidth=2.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

def add_branch_label(ax, x, y, label, color):
    """Add a branch label box."""
    box = FancyBboxPatch((x - 0.7, y - 0.3), 1.4, 0.6,
                        boxstyle="round,pad=0.1",
                        facecolor=color,
                        edgecolor='black',
                        linewidth=2.5,
                        zorder=5)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=6)

def add_arrow(ax, x1, y1, x2, y2, style='solid', color='black'):
    """Add an arrow."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->',
                          mutation_scale=25,
                          linewidth=2.5,
                          color=color,
                          linestyle=style,
                          zorder=2)
    ax.add_patch(arrow)

def create_diagram():
    fig, ax = plt.subplots(figsize=(16, 11), dpi=200)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Title
    ax.text(8, 10.3, 'Branch Creation Workflow', fontsize=20, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#e0e0e0', edgecolor='black', linewidth=2))

    # === STATE 1: Initial (top) ===
    ax.text(8, 9.3, '1. Initial State', fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d0e0f0'))

    # Commits
    add_commit_node(ax, 4, 8, 'C1:\nAdd\nmodel', COLORS['commit'], size=0.8)
    add_commit_node(ax, 7, 8, 'C2:\nAdd\ntests', COLORS['commit'], size=0.8)

    # Arrows between commits
    add_arrow(ax, 6.2, 8, 4.8, 8)

    # Branch label
    add_branch_label(ax, 10, 8, 'main', COLORS['branch_main'])
    add_arrow(ax, 9.3, 8, 7.8, 8, style='dashed', color='gray')

    # HEAD label
    add_branch_label(ax, 13, 8, 'HEAD', COLORS['head'])
    add_arrow(ax, 12.3, 8, 10.7, 8, style='dashed', color='gray')

    # Divider
    ax.plot([0.5, 15.5], [6.8, 6.8], 'k-', linewidth=2, alpha=0.3)

    # === STATE 2: After git branch feature (middle) ===
    ax.text(8, 6.3, '2. After: git branch feature', fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0e0d0'))

    # Commits
    add_commit_node(ax, 4, 5, 'C1:\nAdd\nmodel', COLORS['commit'], size=0.8)
    add_commit_node(ax, 7, 5, 'C2:\nAdd\ntests', COLORS['commit'], size=0.8)

    # Arrows between commits
    add_arrow(ax, 6.2, 5, 4.8, 5)

    # Branch labels (both point to C2)
    add_branch_label(ax, 10, 5.5, 'main', COLORS['branch_main'])
    add_arrow(ax, 9.3, 5.5, 7.8, 5.2, style='dashed', color='gray')

    add_branch_label(ax, 10, 4.5, 'feature', COLORS['branch_feature'])
    add_arrow(ax, 9.3, 4.5, 7.8, 4.8, style='dashed', color='gray')

    # HEAD still points to main
    add_branch_label(ax, 13, 5, 'HEAD', COLORS['head'])
    add_arrow(ax, 12.3, 5, 10.7, 5.5, style='dashed', color='gray')

    ax.text(12, 3.8, 'New branch created,\nHEAD still on main', fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc'))

    # Divider
    ax.plot([0.5, 15.5], [3.5, 3.5], 'k-', linewidth=2, alpha=0.3)

    # === STATE 3: After git checkout feature (bottom) ===
    ax.text(8, 3, '3. After: git checkout feature', fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d0f0d0'))

    # Commits
    add_commit_node(ax, 4, 1.8, 'C1:\nAdd\nmodel', COLORS['commit'], size=0.8)
    add_commit_node(ax, 7, 1.8, 'C2:\nAdd\ntests', COLORS['commit'], size=0.8)

    # Arrows between commits
    add_arrow(ax, 6.2, 1.8, 4.8, 1.8)

    # Branch labels
    add_branch_label(ax, 10, 2.3, 'main', COLORS['branch_feature'])
    add_arrow(ax, 9.3, 2.3, 7.8, 2, style='dashed', color='gray')

    add_branch_label(ax, 10, 1.3, 'feature', COLORS['branch_main'])
    add_arrow(ax, 9.3, 1.3, 7.8, 1.6, style='dashed', color='gray')

    # HEAD now points to feature
    add_branch_label(ax, 13, 1.8, 'HEAD', COLORS['head'])
    add_arrow(ax, 12.3, 1.8, 10.7, 1.3, style='dashed', color='gray')

    ax.text(12, 0.7, 'HEAD switched to feature,\nready to work!', fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc'))

    # Source attribution
    ax.text(0.5, 0.02, 'Generated by: diagram-generators/git_branch_creation_workflow.py',
            fontsize=9, style='italic', color='#666', transform=ax.transAxes)

    # Save
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Generated: {output_path}")

if __name__ == "__main__":
    create_diagram()

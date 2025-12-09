#!/usr/bin/env python3
"""
Generate RAG (Retrieval-Augmented Generation) Architecture diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import os

# Output configuration
OUTPUT_DIR = "../figures"
OUTPUT_FILE = "rag_architecture.png"

# Color scheme
COLORS = {
    'documents': '#e1f5ff',
    'vector_db': '#fff4e1',
    'query': '#ffe1f5',
    'llm': '#e8f5e9',
    'answer': '#c8e6c9',
}

def add_box(ax, x, y, width, height, text, color, fontsize=12, shape='round'):
    """Add a box with text."""
    if shape == 'database':
        # Cylindrical database shape
        ellipse_top = Ellipse((x + width/2, y + height), width/2, height*0.15,
                            facecolor=color, edgecolor='black', linewidth=2.5, zorder=3)
        ax.add_patch(ellipse_top)

        rect = FancyBboxPatch((x, y), width, height,
                            boxstyle="square,pad=0",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2.5,
                            zorder=2)
        ax.add_patch(rect)

        ellipse_bottom = Ellipse((x + width/2, y), width/2, height*0.15,
                              facecolor=color, edgecolor='black', linewidth=2.5, zorder=3)
        ax.add_patch(ellipse_bottom)

        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', zorder=4)
    else:
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2.5)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

def add_arrow(ax, x1, y1, x2, y2, label=''):
    """Add an arrow with optional label."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->,head_width=0.5,head_length=0.7',
                          mutation_scale=25,
                          linewidth=2.5,
                          color='black',
                          zorder=2)
    ax.add_patch(arrow)

    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.3, label, fontsize=10, ha='center', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

def create_diagram():
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.3, 'RAG Architecture: Retrieval-Augmented Generation', fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e0e0e0', edgecolor='black', linewidth=2))

    # Two main sections: Indexing (left) and Query (right)

    # === INDEXING PIPELINE (Left Side) ===
    ax.text(3, 8.3, 'Indexing Pipeline', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#d0d0e0'))

    # Documents
    add_box(ax, 1.5, 7, 3, 0.8, 'Documents', COLORS['documents'])

    # Split into Chunks
    add_box(ax, 1.5, 5.7, 3, 0.8, 'Split into\nChunks', COLORS['documents'])
    add_arrow(ax, 3, 7, 3, 6.5)

    # Generate Embeddings
    add_box(ax, 1.5, 4.4, 3, 0.8, 'Generate\nEmbeddings', COLORS['documents'])
    add_arrow(ax, 3, 5.7, 3, 5.2)

    # Vector Database (center)
    add_box(ax, 5.5, 3.5, 3, 1.2, 'Vector\nDatabase', COLORS['vector_db'], shape='database')
    add_arrow(ax, 4.5, 4.4, 5.5, 4.1)

    # === QUERY PIPELINE (Right Side) ===
    ax.text(11, 8.3, 'Query Pipeline', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0d0d0'))

    # User Query
    add_box(ax, 9.5, 7, 3, 0.8, 'User Query', COLORS['query'])

    # Embed Query
    add_box(ax, 9.5, 5.7, 3, 0.8, 'Embed Query', COLORS['query'])
    add_arrow(ax, 11, 7, 11, 6.5)

    # Search Vector DB
    add_box(ax, 9.5, 4.4, 3, 0.8, 'Search\nVector DB', COLORS['query'])
    add_arrow(ax, 11, 5.7, 11, 5.2)

    # Arrow from Vector DB to Search
    add_arrow(ax, 8.5, 4.1, 9.5, 4.8)

    # Top-K Chunks
    add_box(ax, 9.5, 3.1, 3, 0.8, 'Top-K\nChunks', COLORS['query'])
    add_arrow(ax, 11, 4.4, 11, 3.9)

    # === GENERATION (Bottom Center) ===

    # Augment Prompt
    add_box(ax, 5.5, 1.8, 3, 0.8, 'Augment\nPrompt', COLORS['llm'])
    add_arrow(ax, 11, 3.1, 8.5, 2.2)

    # LLM Generation
    add_box(ax, 5.5, 0.5, 3, 0.8, 'LLM\nGeneration', COLORS['llm'])
    add_arrow(ax, 7, 1.8, 7, 1.3)

    # Answer (with slight offset to the right)
    add_box(ax, 9.5, 0.5, 3, 0.8, 'Answer', COLORS['answer'])
    add_arrow(ax, 8.5, 0.9, 9.5, 0.9)

    # Add descriptive labels
    ax.text(3, 2.7, '1. Index documents offline', fontsize=11, style='italic', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4f8', linewidth=1.5))

    ax.text(11, 2.3, '2. Query at runtime', fontsize=11, style='italic', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8e8f4', linewidth=1.5))

    ax.text(7, 0.1, '3. Generate answer with retrieved context', fontsize=11, style='italic', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8e8', linewidth=1.5))

    # Source attribution
    ax.text(0.5, 0.02, 'Generated by: diagram-generators/rag_architecture.py',
            fontsize=9, style='italic', color='#666', transform=ax.transAxes)

    # Save
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Generated: {output_path}")

if __name__ == "__main__":
    create_diagram()

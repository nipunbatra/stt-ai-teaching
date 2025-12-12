#!/usr/bin/env python3
"""
Generate Git Three States diagram
Shows: Working Directory -> Staging Area -> Repository
"""

from graphviz import Digraph
import os

OUTPUT_DIR = "../figures"
OUTPUT_FILE = "git_three_states"

def create_diagram():
    dot = Digraph(comment='Git Three States', format='png')
    dot.attr(rankdir='LR', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial',
             fontsize='16', width='2.5', height='1.2')
    dot.attr('edge', penwidth='3', fontsize='14', fontname='Arial')

    # Three states with colors
    dot.node('WD', 'Working\\nDirectory', fillcolor='#e1f5ff')
    dot.node('SA', 'Staging Area\\n(Index)', fillcolor='#fff4e1')
    dot.node('REPO', 'Repository\\n(.git)', fillcolor='#e8f5e9')

    # Arrows with labels
    dot.edge('WD', 'SA', label='git add')
    dot.edge('SA', 'REPO', label='git commit')

    # Add title
    dot.attr(label='Git Three States Workflow',
             labelloc='t', fontsize='18', fontname='Arial Bold')

    return dot

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    dot = create_diagram()
    output_path = os.path.join(output_dir, OUTPUT_FILE)
    dot.render(output_path, cleanup=True)
    print(f"✓ Generated: {output_path}.png")

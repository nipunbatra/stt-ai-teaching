#!/usr/bin/env python3
"""
Generate Commit History DAG diagram
Shows: Linear commit history with arrows pointing backwards
"""

from graphviz import Digraph
import os

OUTPUT_DIR = "../figures"
OUTPUT_FILE = "git_commit_dag"

def create_diagram():
    dot = Digraph(comment='Git Commit DAG', format='png')
    dot.attr(rankdir='LR', dpi='300')
    dot.attr('node', shape='circle', style='filled', fillcolor='#e8f5e9',
             fontname='Arial', fontsize='14', width='0.8', height='0.8')
    dot.attr('edge', penwidth='2.5', arrowsize='1.2')

    # Define commits with labels
    commits = [
        ('C0', 'Initial\\ncommit'),
        ('C1', 'Add user\\nmodel'),
        ('C2', 'Add\\nvalidation'),
        ('C3', 'Fix bug'),
        ('C4', 'Add tests'),
    ]

    # Add nodes
    for commit_id, label in commits:
        dot.node(commit_id, commit_id, xlabel=label)

    # Add edges (C1 -> C0, C2 -> C1, etc. - pointing backwards)
    for i in range(len(commits) - 1):
        dot.edge(commits[i+1][0], commits[i][0])

    # Add title and note as a label
    dot.attr(label='Commit History as Directed Acyclic Graph (DAG)\\n' +
                   'Each commit points to its parent (history flows backwards)',
             labelloc='t', fontsize='16', fontname='Arial Bold')

    return dot

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    dot = create_diagram()
    output_path = os.path.join(output_dir, OUTPUT_FILE)
    dot.render(output_path, cleanup=True)
    print(f"✓ Generated: {output_path}.png")

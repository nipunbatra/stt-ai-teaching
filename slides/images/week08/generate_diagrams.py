"""Generate Git concept diagrams for Week 08 slides.

Uses graphviz for graph-based diagrams and matplotlib for
box/flow diagrams. All output as PNG at 200 DPI.
"""

import graphviz
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import subprocess

# ─── Color palette (matches iitgn-modern.css) ──────────────────────────────

PRIMARY = '#1e3a5f'
PRIMARY_LIGHT = '#2e5a8f'
ACCENT = '#e85a4f'
SUCCESS = '#2a9d8f'
WARNING = '#e9c46a'
PURPLE = '#7c3aed'
ORANGE = '#e76f51'
WHITE = '#ffffff'
BG = '#f7fafc'
TEXT = '#2d3748'
TEXT_LIGHT = '#4a5568'

# ─── Helper: graphviz commit node ──────────────────────────────────────────

def commit_node(g, name, label=None, color=PRIMARY, shape='circle'):
    """Add a styled commit node to a graphviz graph."""
    g.node(name, label=label or name,
           shape=shape, style='filled', fillcolor=color,
           fontcolor='white', fontname='Helvetica Bold', fontsize='11',
           width='0.5', height='0.5', fixedsize='true')


def box_node(g, name, label, fillcolor='#e8f4fd', fontcolor=PRIMARY,
             shape='box', style='filled,rounded', **kwargs):
    """Add a styled box node."""
    g.node(name, label=label, shape=shape, style=style,
           fillcolor=fillcolor, fontcolor=fontcolor,
           fontname='Helvetica Bold', fontsize='12',
           margin='0.2,0.1', **kwargs)


def branch_label(g, name, label, color=PRIMARY, bg='#e8f0fe'):
    """Add a branch label node."""
    g.node(name, label=label, shape='box', style='filled,rounded',
           fillcolor=bg, fontcolor=color, fontname='Helvetica Bold',
           fontsize='11', width='0', height='0', margin='0.08,0.04',
           penwidth='1.5', color=color)


# ═══════════════════════════════════════════════════════════════════════════
# 1. THREE AREAS: Working Dir → Staging → Repository
# ═══════════════════════════════════════════════════════════════════════════

def draw_three_areas():
    g = graphviz.Digraph('three_areas', format='png',
                         graph_attr={'rankdir': 'LR', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.5',
                                     'nodesep': '1.0', 'ranksep': '1.5'})

    # Three area boxes with contents
    g.node('wd', label='''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">
        <TR><TD><B>Working Directory</B></TD></TR>
        <TR><TD><FONT FACE="Courier" POINT-SIZE="10">train.py  ✎</FONT></TD></TR>
        <TR><TD><FONT FACE="Courier" POINT-SIZE="10">utils.py  ✎</FONT></TD></TR>
        <TR><TD><FONT FACE="Courier" POINT-SIZE="10">config.yaml</FONT></TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded',
           fillcolor='#e8f4fd', color=PRIMARY_LIGHT, penwidth='2.5',
           fontname='Helvetica', width='2.2', height='1.5')

    g.node('sa', label='''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">
        <TR><TD><B>Staging Area</B></TD></TR>
        <TR><TD><FONT FACE="Courier" POINT-SIZE="10">train.py</FONT></TD></TR>
        <TR><TD><FONT FACE="Courier" POINT-SIZE="10">utils.py</FONT></TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded',
           fillcolor='#fef9e7', color='#d4a017', penwidth='2.5',
           fontname='Helvetica', width='2.0', height='1.5')

    g.node('repo', label='''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">
        <TR><TD><B>Repository (.git)</B></TD></TR>
        <TR><TD><FONT FACE="Courier" POINT-SIZE="10">commit abc1234</FONT></TD></TR>
        <TR><TD><FONT FACE="Courier" POINT-SIZE="10">commit def5678</FONT></TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded',
           fillcolor='#e8f8f5', color=SUCCESS, penwidth='2.5',
           fontname='Helvetica', width='2.2', height='1.5')

    g.edge('wd', 'sa', label='  git add  ', fontname='Courier Bold',
           fontsize='13', fontcolor=ACCENT, color=ACCENT, penwidth='2.5',
           arrowsize='1.2')
    g.edge('sa', 'repo', label='  git commit  ', fontname='Courier Bold',
           fontsize='13', fontcolor=ACCENT, color=ACCENT, penwidth='2.5',
           arrowsize='1.2')

    g.render('git_three_areas', cleanup=True)
    print('  Created git_three_areas.png')


# ═══════════════════════════════════════════════════════════════════════════
# 2. FILE LIFECYCLE: the four states
# ═══════════════════════════════════════════════════════════════════════════

def draw_file_lifecycle():
    g = graphviz.Digraph('file_lifecycle', format='png',
                         graph_attr={'rankdir': 'LR', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.4',
                                     'nodesep': '0.8', 'ranksep': '1.2'})

    states = [
        ('untracked', 'Untracked', '#fef2f2', ACCENT),
        ('staged',    'Staged',    '#fef9e7', '#d4a017'),
        ('committed', 'Committed', '#e8f8f5', SUCCESS),
        ('modified',  'Modified',  '#eff6ff', PRIMARY_LIGHT),
    ]
    for sid, label, bg, color in states:
        g.node(sid, label=label, shape='box', style='filled,rounded',
               fillcolor=bg, color=color, fontcolor=color, penwidth='2',
               fontname='Helvetica Bold', fontsize='13',
               width='1.4', height='0.6')

    edges = [
        ('untracked', 'staged',    'git add'),
        ('staged',    'committed', 'git commit'),
        ('committed', 'modified',  'edit file'),
        ('modified',  'staged',    'git add'),
    ]
    for src, dst, label in edges:
        g.edge(src, dst, label=f'  {label}  ', fontname='Courier Bold',
               fontsize='10', fontcolor=TEXT_LIGHT, color=TEXT,
               penwidth='1.5', arrowsize='0.9')

    g.render('git_file_lifecycle', cleanup=True)
    print('  Created git_file_lifecycle.png')


# ═══════════════════════════════════════════════════════════════════════════
# 3. BRANCH & MERGE (three-way merge with divergence)
# ═══════════════════════════════════════════════════════════════════════════

def draw_branch_merge():
    g = graphviz.Digraph('branch_merge', format='png',
                         graph_attr={'rankdir': 'LR', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.4',
                                     'nodesep': '0.4', 'ranksep': '0.7'})

    # Main branch commits
    for c in ['c1', 'c2', 'c3', 'c5']:
        commit_node(g, c, color=PRIMARY)
    commit_node(g, 'M', label='M', color=ACCENT)

    # Feature branch commits
    commit_node(g, 'c4a', color=SUCCESS)
    commit_node(g, 'c4b', color=SUCCESS)

    # Main edges
    for a, b in [('c1','c2'), ('c2','c3'), ('c3','c5'), ('c5','M')]:
        g.edge(a, b, color=PRIMARY, penwidth='2.5', arrowsize='0.8')

    # Feature edges
    g.edge('c2', 'c4a', color=SUCCESS, penwidth='2.5', arrowsize='0.8')
    g.edge('c4a', 'c4b', color=SUCCESS, penwidth='2.5', arrowsize='0.8')
    g.edge('c4b', 'M', color=SUCCESS, penwidth='2', style='dashed', arrowsize='0.8')

    # Branch labels (invisible edges for positioning)
    branch_label(g, 'main_lbl', 'main', PRIMARY, '#e8f0fe')
    branch_label(g, 'feat_lbl', 'feature/augment', SUCCESS, '#e8f8f5')
    g.edge('M', 'main_lbl', style='dotted', color='#cccccc', arrowsize='0.5')
    g.edge('c4b', 'feat_lbl', style='dotted', color='#cccccc', arrowsize='0.5')

    # Align main branch horizontally
    with g.subgraph() as s:
        s.attr(rank='same')
        s.node('c1'); s.node('c2'); s.node('c3'); s.node('c5'); s.node('M')

    with g.subgraph() as s:
        s.attr(rank='same')
        s.node('c4a'); s.node('c4b')

    g.render('git_branch_merge', cleanup=True)
    print('  Created git_branch_merge.png')


# ═══════════════════════════════════════════════════════════════════════════
# 4. FAST-FORWARD vs THREE-WAY MERGE (side by side)
# ═══════════════════════════════════════════════════════════════════════════

def draw_ff_vs_threeway():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(WHITE)

    r = 0.25

    def draw_c(ax, x, y, label, color, radius=0.25):
        circle = plt.Circle((x, y), radius, color=color, ec='white', lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=4, family='monospace')

    # ── Left: Fast-Forward ──
    ax1.set_xlim(-0.5, 8); ax1.set_ylim(-0.5, 4)
    ax1.set_aspect('equal'); ax1.axis('off')
    ax1.set_title('Fast-Forward Merge', fontsize=14, fontweight='bold',
                  color=PRIMARY, pad=10)

    # Before
    ax1.text(2.0, 3.5, 'Before:', fontsize=11, fontweight='bold', color=TEXT)
    for i, (x, lbl) in enumerate([(1,  'c1'), (2.5, 'c2')]):
        draw_c(ax1, x, 3.0, lbl, PRIMARY, r)
    ax1.plot([1+r, 2.5-r], [3, 3], color=PRIMARY, lw=2.5)
    draw_c(ax1, 4.0, 3.0, 'c3', SUCCESS, r)
    ax1.plot([2.5+r, 4.0-r], [3, 3], color=SUCCESS, lw=2.5)
    ax1.text(2.5, 2.4, 'main', ha='center', fontsize=9, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1))
    ax1.text(4.0, 2.4, 'feature', ha='center', fontsize=9, fontweight='bold',
             color=SUCCESS, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f8f5', ec=SUCCESS, lw=1))

    # After
    ax1.text(2.0, 1.5, 'After:', fontsize=11, fontweight='bold', color=TEXT)
    for i, (x, lbl) in enumerate([(1, 'c1'), (2.5, 'c2'), (4.0, 'c3')]):
        draw_c(ax1, x, 1.0, lbl, PRIMARY, r)
    ax1.plot([1+r, 2.5-r], [1, 1], color=PRIMARY, lw=2.5)
    ax1.plot([2.5+r, 4.0-r], [1, 1], color=PRIMARY, lw=2.5)
    ax1.text(4.0, 0.4, 'main', ha='center', fontsize=9, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1))
    ax1.text(5.5, 1.0, 'just moves\nthe pointer!', ha='center', fontsize=10,
             color=SUCCESS, style='italic', fontweight='bold')

    # ── Right: Three-Way Merge ──
    ax2.set_xlim(-0.5, 9); ax2.set_ylim(-0.5, 4)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('Three-Way Merge', fontsize=14, fontweight='bold',
                  color=PRIMARY, pad=10)

    # Before
    ax2.text(2.5, 3.5, 'Before:', fontsize=11, fontweight='bold', color=TEXT)
    draw_c(ax2, 1, 3.0, 'c1', PRIMARY, r)
    draw_c(ax2, 2.5, 3.0, 'c2', PRIMARY, r); draw_c(ax2, 4.0, 3.0, 'c3', PRIMARY, r)
    draw_c(ax2, 3.5, 2.0, 'c4', SUCCESS, r)
    ax2.plot([1+r, 2.5-r], [3, 3], color=PRIMARY, lw=2.5)
    ax2.plot([2.5+r, 4.0-r], [3, 3], color=PRIMARY, lw=2.5)
    ax2.plot([2.5+0.15, 3.5-r], [3-0.15, 2.0+0.1], color=SUCCESS, lw=2.5)
    ax2.text(4.0, 3.6, 'main', ha='center', fontsize=9, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1))
    ax2.text(3.5, 1.4, 'feature', ha='center', fontsize=9, fontweight='bold',
             color=SUCCESS, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f8f5', ec=SUCCESS, lw=1))

    # After
    ax2.text(2.5, 0.7, 'After:', fontsize=11, fontweight='bold', color=TEXT)
    draw_c(ax2, 1.0, 0.0, 'c1', PRIMARY, r)
    draw_c(ax2, 2.5, 0.0, 'c2', PRIMARY, r); draw_c(ax2, 4.0, 0.0, 'c3', PRIMARY, r)
    draw_c(ax2, 3.5, -0.8, 'c4', SUCCESS, r)
    draw_c(ax2, 5.5, 0.0, 'M', ACCENT, r)
    ax2.plot([1+r, 2.5-r], [0, 0], color=PRIMARY, lw=2.5)
    ax2.plot([2.5+r, 4.0-r], [0, 0], color=PRIMARY, lw=2.5)
    ax2.plot([4.0+r, 5.5-r], [0, 0], color=PRIMARY, lw=2.5)
    ax2.plot([2.5+0.15, 3.5-r], [0-0.15, -0.8+0.1], color=SUCCESS, lw=2.5)
    ax2.plot([3.5+0.15, 5.5-0.15], [-0.8+0.15, 0-0.1], color=SUCCESS, lw=2, ls='--')
    ax2.text(5.5, -0.65, 'main', ha='center', fontsize=9, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1))
    ax2.text(7.0, 0.0, 'new merge\ncommit!', ha='center', fontsize=10,
             color=ACCENT, style='italic', fontweight='bold')

    plt.tight_layout(w_pad=3)
    fig.savefig('git_ff_vs_threeway.png', dpi=200, bbox_inches='tight',
                facecolor=WHITE, edgecolor='none')
    plt.close()
    print('  Created git_ff_vs_threeway.png')


# ═══════════════════════════════════════════════════════════════════════════
# 5. FEATURE BRANCH WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════

def draw_feature_workflow():
    g = graphviz.Digraph('feature_workflow', format='png',
                         graph_attr={'rankdir': 'LR', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.5',
                                     'nodesep': '0.3', 'ranksep': '0.6',
                                     'label': 'Feature Branch Workflow',
                                     'labelloc': 't', 'fontsize': '18',
                                     'fontname': 'Helvetica Bold',
                                     'fontcolor': PRIMARY})

    # Main branch
    for c in ['m1','m2','m3','m4','m5','m6']:
        commit_node(g, c, label='', color=PRIMARY)
    for a, b in [('m1','m2'),('m2','m3'),('m3','m4'),('m4','m5'),('m5','m6')]:
        g.edge(a, b, color=PRIMARY, penwidth='3', arrowsize='0.7')

    # Feature 1 (green)
    for c in ['f1a','f1b']:
        commit_node(g, c, label='', color=SUCCESS)
    g.edge('m1', 'f1a', color=SUCCESS, penwidth='2', arrowsize='0.7')
    g.edge('f1a', 'f1b', color=SUCCESS, penwidth='2', arrowsize='0.7')
    g.edge('f1b', 'm3', color=SUCCESS, penwidth='1.5', style='dashed', arrowsize='0.7')

    # Feature 2 (orange)
    for c in ['f2a','f2b']:
        commit_node(g, c, label='', color=ORANGE)
    g.edge('m3', 'f2a', color=ORANGE, penwidth='2', arrowsize='0.7')
    g.edge('f2a', 'f2b', color=ORANGE, penwidth='2', arrowsize='0.7')
    g.edge('f2b', 'm5', color=ORANGE, penwidth='1.5', style='dashed', arrowsize='0.7')

    # Feature 3 (purple)
    for c in ['f3a','f3b']:
        commit_node(g, c, label='', color=PURPLE)
    g.edge('m4', 'f3a', color=PURPLE, penwidth='2', arrowsize='0.7')
    g.edge('f3a', 'f3b', color=PURPLE, penwidth='2', arrowsize='0.7')
    g.edge('f3b', 'm6', color=PURPLE, penwidth='1.5', style='dashed', arrowsize='0.7')

    # Branch labels
    branch_label(g, 'main_l', 'main', PRIMARY, '#e8f0fe')
    branch_label(g, 'f1_l', 'feature/login', SUCCESS, '#e8f8f5')
    branch_label(g, 'f2_l', 'feature/search', ORANGE, '#fef2e8')
    branch_label(g, 'f3_l', 'feature/dashboard', PURPLE, '#f3e8ff')

    g.edge('m6', 'main_l', style='dotted', color='#cccccc', arrowsize='0.4')
    g.edge('f1b', 'f1_l', style='dotted', color='#cccccc', arrowsize='0.4')
    g.edge('f2b', 'f2_l', style='dotted', color='#cccccc', arrowsize='0.4')
    g.edge('f3b', 'f3_l', style='dotted', color='#cccccc', arrowsize='0.4')

    # PR labels
    for name, label, color in [('pr1','PR #1',SUCCESS),('pr2','PR #2',ORANGE),('pr3','PR #3',PURPLE)]:
        g.node(name, label=label, shape='note', style='filled', fillcolor='white',
               fontcolor=color, color=color, fontname='Helvetica Bold',
               fontsize='10', width='0.6', height='0.3')
    g.edge('f1b', 'pr1', style='invis')
    g.edge('f2b', 'pr2', style='invis')
    g.edge('f3b', 'pr3', style='invis')

    # Rank alignment
    with g.subgraph() as s:
        s.attr(rank='same')
        for n in ['m1','m2','m3','m4','m5','m6']:
            s.node(n)

    g.render('git_feature_workflow', cleanup=True)
    print('  Created git_feature_workflow.png')


# ═══════════════════════════════════════════════════════════════════════════
# 6. RESET: --soft vs --mixed vs --hard
# ═══════════════════════════════════════════════════════════════════════════

def draw_reset_modes():
    g = graphviz.Digraph('reset_modes', format='png',
                         graph_attr={'rankdir': 'TB', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.5',
                                     'nodesep': '0.6', 'ranksep': '0.8',
                                     'label': 'git reset: Three Modes',
                                     'labelloc': 't', 'fontsize': '16',
                                     'fontname': 'Helvetica Bold',
                                     'fontcolor': PRIMARY})

    # Before state
    g.node('before', label='''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="3" BGCOLOR="#e8f4fd"><B>Before: git reset HEAD~1</B></TD></TR>
        <TR><TD BGCOLOR="#e8f8f5">Working Dir<BR/><FONT POINT-SIZE="9">changed file</FONT></TD>
            <TD BGCOLOR="#fef9e7">Staging<BR/><FONT POINT-SIZE="9">changed file</FONT></TD>
            <TD BGCOLOR="#e8f4fd">Repo<BR/><FONT POINT-SIZE="9">commit C</FONT></TD></TR>
    </TABLE>>''', shape='none')

    # Three outcomes
    g.node('soft', label='''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="3" BGCOLOR="#e8f8f5"><B>--soft</B></TD></TR>
        <TR><TD BGCOLOR="#e8f8f5">Working Dir<BR/><FONT POINT-SIZE="9">changed &#10004;</FONT></TD>
            <TD BGCOLOR="#fef9e7">Staging<BR/><FONT POINT-SIZE="9">changed &#10004;</FONT></TD>
            <TD BGCOLOR="#fef2f2">Repo<BR/><FONT POINT-SIZE="9"><S>commit C</S></FONT></TD></TR>
    </TABLE>>''', shape='none')

    g.node('mixed', label='''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="3" BGCOLOR="#fef9e7"><B>--mixed (default)</B></TD></TR>
        <TR><TD BGCOLOR="#e8f8f5">Working Dir<BR/><FONT POINT-SIZE="9">changed &#10004;</FONT></TD>
            <TD BGCOLOR="#fef2f2">Staging<BR/><FONT POINT-SIZE="9"><S>unstaged</S></FONT></TD>
            <TD BGCOLOR="#fef2f2">Repo<BR/><FONT POINT-SIZE="9"><S>commit C</S></FONT></TD></TR>
    </TABLE>>''', shape='none')

    g.node('hard', label='''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="3" BGCOLOR="#fef2f2"><B>--hard  &#9888; DANGER</B></TD></TR>
        <TR><TD BGCOLOR="#fef2f2">Working Dir<BR/><FONT POINT-SIZE="9"><S>GONE</S></FONT></TD>
            <TD BGCOLOR="#fef2f2">Staging<BR/><FONT POINT-SIZE="9"><S>GONE</S></FONT></TD>
            <TD BGCOLOR="#fef2f2">Repo<BR/><FONT POINT-SIZE="9"><S>commit C</S></FONT></TD></TR>
    </TABLE>>''', shape='none')

    g.edge('before', 'soft', label='  keeps everything  ', color=SUCCESS,
           fontcolor=SUCCESS, fontname='Helvetica Bold', fontsize='10', penwidth='2')
    g.edge('before', 'mixed', label='  unstages changes  ', color=WARNING,
           fontcolor='#b8860b', fontname='Helvetica Bold', fontsize='10', penwidth='2')
    g.edge('before', 'hard', label='  deletes everything  ', color=ACCENT,
           fontcolor=ACCENT, fontname='Helvetica Bold', fontsize='10', penwidth='2')

    g.render('git_reset_modes', cleanup=True)
    print('  Created git_reset_modes.png')


# ═══════════════════════════════════════════════════════════════════════════
# 7. LOCAL vs REMOTE
# ═══════════════════════════════════════════════════════════════════════════

def draw_local_remote():
    g = graphviz.Digraph('local_remote', format='png',
                         graph_attr={'rankdir': 'LR', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.5',
                                     'compound': 'true'})

    # Local cluster
    with g.subgraph(name='cluster_local') as c:
        c.attr(label='Your Machine', style='rounded,filled',
               fillcolor='#f0f4f8', color=PRIMARY, fontname='Helvetica Bold',
               fontsize='14', fontcolor=PRIMARY, penwidth='2')
        c.node('wd', label='Working\nDirectory', shape='folder',
               style='filled', fillcolor='#e8f4fd', color=PRIMARY_LIGHT,
               fontname='Helvetica', fontsize='11', fontcolor=PRIMARY)
        c.node('stage', label='Staging\nArea', shape='tab',
               style='filled', fillcolor='#fef9e7', color='#d4a017',
               fontname='Helvetica', fontsize='11', fontcolor='#b8860b')
        c.node('local', label='Local\nRepository', shape='cylinder',
               style='filled', fillcolor='#e8f8f5', color=SUCCESS,
               fontname='Helvetica', fontsize='11', fontcolor=SUCCESS)

    # Remote cluster
    with g.subgraph(name='cluster_remote') as c:
        c.attr(label='GitHub / GitLab', style='rounded,filled',
               fillcolor='#f0f4f8', color=ORANGE, fontname='Helvetica Bold',
               fontsize='14', fontcolor=ORANGE, penwidth='2')
        c.node('remote', label='Remote\nRepository\n(origin)', shape='cylinder',
               style='filled', fillcolor='#fef2e8', color=ORANGE,
               fontname='Helvetica', fontsize='11', fontcolor=ORANGE)

    # Edges
    g.edge('wd', 'stage', label=' git add ', fontname='Courier Bold',
           fontsize='10', fontcolor=ACCENT, color=ACCENT, penwidth='2')
    g.edge('stage', 'local', label=' git commit ', fontname='Courier Bold',
           fontsize='10', fontcolor=ACCENT, color=ACCENT, penwidth='2')
    g.edge('local', 'remote', label=' git push ', fontname='Courier Bold',
           fontsize='10', fontcolor=SUCCESS, color=SUCCESS, penwidth='2.5')
    g.edge('remote', 'local', label=' git pull / fetch ', fontname='Courier Bold',
           fontsize='10', fontcolor=PRIMARY_LIGHT, color=PRIMARY_LIGHT,
           penwidth='2.5', style='dashed')

    g.render('git_local_remote', cleanup=True)
    print('  Created git_local_remote.png')


# ═══════════════════════════════════════════════════════════════════════════
# 8. MERGE CONFLICT RESOLUTION FLOW
# ═══════════════════════════════════════════════════════════════════════════

def draw_conflict_flow():
    g = graphviz.Digraph('conflict_flow', format='png',
                         graph_attr={'rankdir': 'TB', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.4',
                                     'nodesep': '0.4', 'ranksep': '0.5',
                                     'label': 'Resolving a Merge Conflict',
                                     'labelloc': 't', 'fontsize': '16',
                                     'fontname': 'Helvetica Bold',
                                     'fontcolor': PRIMARY})

    steps = [
        ('s1', 'git merge feature', '#e8f4fd', PRIMARY_LIGHT),
        ('s2', 'CONFLICT!\nAuto-merge failed', '#fef2f2', ACCENT),
        ('s3', 'Open file, find\n<<<<<<< markers', '#fef9e7', '#d4a017'),
        ('s4', 'Choose correct version\n(or combine both)', '#eff6ff', PRIMARY_LIGHT),
        ('s5', 'Remove conflict markers\n<<<, ===, >>>', '#fef9e7', '#d4a017'),
        ('s6', 'git add <file>', '#e8f8f5', SUCCESS),
        ('s7', 'git commit', '#e8f8f5', SUCCESS),
    ]

    for sid, label, bg, color in steps:
        shape = 'diamond' if sid == 's2' else 'box'
        g.node(sid, label=label, shape=shape, style='filled,rounded',
               fillcolor=bg, color=color, fontcolor=TEXT,
               fontname='Helvetica', fontsize='11',
               width='2.5' if shape == 'box' else '2.0')

    for a, b in [('s1','s2'),('s2','s3'),('s3','s4'),('s4','s5'),('s5','s6'),('s6','s7')]:
        g.edge(a, b, color=TEXT_LIGHT, penwidth='1.5', arrowsize='0.8')

    g.render('git_conflict_flow', cleanup=True)
    print('  Created git_conflict_flow.png')


# ═══════════════════════════════════════════════════════════════════════════
# 9. STASH WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════

def draw_stash_workflow():
    g = graphviz.Digraph('stash_workflow', format='png',
                         graph_attr={'rankdir': 'LR', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.5',
                                     'nodesep': '0.7', 'ranksep': '1.2',
                                     'label': 'Git Stash: Your Pocket',
                                     'labelloc': 't', 'fontsize': '16',
                                     'fontname': 'Helvetica Bold',
                                     'fontcolor': PRIMARY})

    g.node('dirty', label='Dirty Working\nDirectory\n(WIP changes)', shape='box',
           style='filled,rounded', fillcolor='#fef9e7', color='#d4a017',
           fontname='Helvetica', fontsize='11', fontcolor=TEXT, width='1.8')

    g.node('clean', label='Clean Working\nDirectory', shape='box',
           style='filled,rounded', fillcolor='#e8f8f5', color=SUCCESS,
           fontname='Helvetica', fontsize='11', fontcolor=TEXT, width='1.8')

    g.node('stash', label='Stash Stack\n(hidden pocket)', shape='box3d',
           style='filled', fillcolor='#eff6ff', color=PRIMARY_LIGHT,
           fontname='Helvetica', fontsize='11', fontcolor=PRIMARY, width='1.8')

    g.node('other', label='Switch branch\nFix urgent bug\nDo other work', shape='note',
           style='filled', fillcolor='#f7fafc', color=TEXT_LIGHT,
           fontname='Helvetica', fontsize='10', fontcolor=TEXT_LIGHT, width='1.8')

    g.edge('dirty', 'clean', label='  git stash  ', fontname='Courier Bold',
           fontsize='11', fontcolor=ACCENT, color=ACCENT, penwidth='2.5')
    g.edge('dirty', 'stash', label='  saves to  ', fontname='Helvetica',
           fontsize='10', fontcolor=TEXT_LIGHT, color=TEXT_LIGHT,
           penwidth='1.5', style='dashed')
    g.edge('clean', 'other', label='  free to work  ', fontname='Helvetica',
           fontsize='10', fontcolor=TEXT_LIGHT, color=TEXT_LIGHT, penwidth='1.5')
    g.edge('stash', 'dirty', label='  git stash pop  ', fontname='Courier Bold',
           fontsize='11', fontcolor=SUCCESS, color=SUCCESS, penwidth='2.5',
           constraint='false')

    g.render('git_stash_workflow', cleanup=True)
    print('  Created git_stash_workflow.png')


# ═══════════════════════════════════════════════════════════════════════════
# 10. COMMIT ANATOMY (what's inside a commit)
# ═══════════════════════════════════════════════════════════════════════════

def draw_commit_anatomy():
    g = graphviz.Digraph('commit_anatomy', format='png',
                         graph_attr={'rankdir': 'LR', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.5',
                                     'nodesep': '0.5', 'ranksep': '1.0'})

    # Commit object
    g.node('commit', label='''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD COLSPAN="2" BGCOLOR="#e8f4fd"><B>Commit def5678</B></TD></TR>
        <TR><TD ALIGN="LEFT">tree</TD><TD ALIGN="LEFT">→ snapshot of files</TD></TR>
        <TR><TD ALIGN="LEFT">parent</TD><TD ALIGN="LEFT">→ abc1234</TD></TR>
        <TR><TD ALIGN="LEFT">author</TD><TD ALIGN="LEFT">Nipun Batra</TD></TR>
        <TR><TD ALIGN="LEFT">date</TD><TD ALIGN="LEFT">2026-02-28</TD></TR>
        <TR><TD ALIGN="LEFT">message</TD><TD ALIGN="LEFT">"Add LR scheduler"</TD></TR>
    </TABLE>>''', shape='none')

    # Tree (snapshot)
    g.node('tree', label='''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#e8f8f5"><B>Tree (Snapshot)</B></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="10">train.py   → blob a3f...</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="10">utils.py   → blob 7c2...</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="10">config.yaml → blob e1d...</FONT></TD></TR>
    </TABLE>>''', shape='none')

    # Parent commit
    g.node('parent', label='''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD BGCOLOR="#fef9e7"><B>Parent abc1234</B></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10">previous commit...</FONT></TD></TR>
    </TABLE>>''', shape='none')

    g.edge('commit', 'tree', label='  tree pointer  ', fontname='Helvetica',
           fontsize='9', fontcolor=SUCCESS, color=SUCCESS, penwidth='2')
    g.edge('commit', 'parent', label='  parent pointer  ', fontname='Helvetica',
           fontsize='9', fontcolor='#d4a017', color='#d4a017', penwidth='2')

    g.render('git_commit_anatomy', cleanup=True)
    print('  Created git_commit_anatomy.png')


# ═══════════════════════════════════════════════════════════════════════════
# 11. SNAPSHOT MODEL (two commits side by side)
# ═══════════════════════════════════════════════════════════════════════════

def draw_snapshot_model():
    g = graphviz.Digraph('snapshot_model', format='png',
                         graph_attr={'rankdir': 'LR', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.5',
                                     'nodesep': '0.8', 'ranksep': '1.2'})

    g.node('c1', label='''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="2" BGCOLOR="#e8f4fd"><B>Commit 1 (abc1234)</B></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="10">train.py</FONT></TD>
            <TD ALIGN="LEFT"><FONT POINT-SIZE="10">v1</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="10">utils.py</FONT></TD>
            <TD ALIGN="LEFT"><FONT POINT-SIZE="10">v1</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="10">config.yaml</FONT></TD>
            <TD ALIGN="LEFT"><FONT POINT-SIZE="10">v1</FONT></TD></TR>
    </TABLE>>''', shape='none')

    g.node('c2', label='''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="2" BGCOLOR="#e8f8f5"><B>Commit 2 (def5678)</B></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="10">train.py</FONT></TD>
            <TD ALIGN="LEFT" BGCOLOR="#fef2f2"><FONT POINT-SIZE="10"><B>v2 changed!</B></FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="10">utils.py</FONT></TD>
            <TD ALIGN="LEFT"><FONT POINT-SIZE="10">→ v1 (pointer)</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="10">config.yaml</FONT></TD>
            <TD ALIGN="LEFT"><FONT POINT-SIZE="10">→ v1 (pointer)</FONT></TD></TR>
    </TABLE>>''', shape='none')

    g.edge('c1', 'c2', label='  parent →  ', fontname='Helvetica Bold',
           fontsize='11', fontcolor=PRIMARY, color=PRIMARY, penwidth='2.5',
           arrowsize='1.2')

    g.render('git_snapshot_model', cleanup=True)
    print('  Created git_snapshot_model.png')


# ═══════════════════════════════════════════════════════════════════════════
# 12. COMMIT CHAIN with HEAD
# ═══════════════════════════════════════════════════════════════════════════

def draw_commit_chain():
    g = graphviz.Digraph('commit_chain', format='png',
                         graph_attr={'rankdir': 'LR', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.4',
                                     'nodesep': '0.5', 'ranksep': '0.7'})

    for c in ['c1', 'c2', 'c3', 'c4']:
        commit_node(g, c, color=PRIMARY)
    for a, b in [('c1','c2'), ('c2','c3'), ('c3','c4')]:
        g.edge(a, b, color=PRIMARY, penwidth='2.5', arrowsize='0.8')

    # HEAD label pointing at c4
    g.node('head', label='HEAD', shape='box', style='filled,rounded',
           fillcolor='#fef2f2', fontcolor=ACCENT, color=ACCENT,
           fontname='Courier Bold', fontsize='12',
           width='0.8', height='0.3', penwidth='2')
    branch_label(g, 'main_l', 'main', PRIMARY, '#e8f0fe')

    g.edge('head', 'c4', style='dashed', color=ACCENT, penwidth='1.5', arrowsize='0.7')
    g.edge('main_l', 'c4', style='dotted', color='#cccccc', arrowsize='0.5')

    with g.subgraph() as s:
        s.attr(rank='same')
        s.node('c1'); s.node('c2'); s.node('c3'); s.node('c4')

    g.render('git_commit_chain', cleanup=True)
    print('  Created git_commit_chain.png')


# ═══════════════════════════════════════════════════════════════════════════
# 13. BRANCH AS POINTER (before/after creating branch)
# ═══════════════════════════════════════════════════════════════════════════

def draw_branch_pointer():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    fig.patch.set_facecolor(WHITE)
    r = 0.25

    def draw_c(ax, x, y, label, color, radius=0.25):
        circle = plt.Circle((x, y), radius, color=color, ec='white', lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=4, family='monospace')

    # ── Top: Before ──
    ax1.set_xlim(-0.5, 9); ax1.set_ylim(-0.8, 1.5)
    ax1.set_aspect('equal'); ax1.axis('off')
    ax1.set_title('Before: only main branch', fontsize=13, fontweight='bold',
                  color=PRIMARY, loc='left', pad=5)

    for i, (x, lbl) in enumerate([(1, 'c1'), (3, 'c2'), (5, 'c3')]):
        draw_c(ax1, x, 0.3, lbl, PRIMARY, r)
    ax1.plot([1+r, 3-r], [0.3, 0.3], color=PRIMARY, lw=2.5)
    ax1.plot([3+r, 5-r], [0.3, 0.3], color=PRIMARY, lw=2.5)

    ax1.text(5, -0.35, 'main', ha='center', fontsize=10, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1.5))
    ax1.text(5, 1.0, 'HEAD', ha='center', fontsize=10, fontweight='bold',
             color=ACCENT, bbox=dict(boxstyle='round,pad=0.15', fc='#fef2f2', ec=ACCENT, lw=1.5))
    ax1.annotate('', xy=(5, 0.55), xytext=(5, 0.85),
                 arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.5))

    # ── Bottom: After ──
    ax2.set_xlim(-0.5, 9); ax2.set_ylim(-0.8, 1.5)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('After: git checkout -b feature', fontsize=13, fontweight='bold',
                  color=SUCCESS, loc='left', pad=5)

    for i, (x, lbl) in enumerate([(1, 'c1'), (3, 'c2'), (5, 'c3')]):
        draw_c(ax2, x, 0.3, lbl, PRIMARY, r)
    ax2.plot([1+r, 3-r], [0.3, 0.3], color=PRIMARY, lw=2.5)
    ax2.plot([3+r, 5-r], [0.3, 0.3], color=PRIMARY, lw=2.5)

    ax2.text(5.8, -0.35, 'main', ha='center', fontsize=10, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1.5))
    ax2.text(4.2, -0.35, 'feature', ha='center', fontsize=10, fontweight='bold',
             color=SUCCESS, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f8f5', ec=SUCCESS, lw=1.5))
    ax2.text(5, 1.0, 'HEAD', ha='center', fontsize=10, fontweight='bold',
             color=ACCENT, bbox=dict(boxstyle='round,pad=0.15', fc='#fef2f2', ec=ACCENT, lw=1.5))
    ax2.annotate('', xy=(4.5, 0.55), xytext=(5, 0.85),
                 arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.5))

    ax2.text(7.0, 0.3, 'Both point to\nthe same commit!', ha='center', fontsize=11,
             color=SUCCESS, style='italic', fontweight='bold')

    plt.tight_layout(h_pad=1.5)
    fig.savefig('git_branch_pointer.png', dpi=200, bbox_inches='tight',
                facecolor=WHITE, edgecolor='none')
    plt.close()
    print('  Created git_branch_pointer.png')


# ═══════════════════════════════════════════════════════════════════════════
# 14. DIVERGING BRANCHES
# ═══════════════════════════════════════════════════════════════════════════

def draw_diverging_branches():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    fig.patch.set_facecolor(WHITE)
    r = 0.25

    def draw_c(ax, x, y, label, color, radius=0.25):
        circle = plt.Circle((x, y), radius, color=color, ec='white', lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=4, family='monospace')

    # ── Top: Feature branch has a commit, main hasn't moved ──
    ax1.set_xlim(-0.5, 10); ax1.set_ylim(-0.5, 2.5)
    ax1.set_aspect('equal'); ax1.axis('off')
    ax1.set_title('Step 1: Commit on feature branch', fontsize=13,
                  fontweight='bold', color=PRIMARY, loc='left', pad=5)

    for x, lbl in [(1, 'c1'), (3, 'c2'), (5, 'c3')]:
        draw_c(ax1, x, 0.5, lbl, PRIMARY, r)
    draw_c(ax1, 7, 1.5, 'A', SUCCESS, r)
    ax1.plot([1+r, 3-r], [0.5, 0.5], color=PRIMARY, lw=2.5)
    ax1.plot([3+r, 5-r], [0.5, 0.5], color=PRIMARY, lw=2.5)
    ax1.plot([5+0.15, 7-r], [0.5+0.15, 1.5], color=SUCCESS, lw=2.5)

    ax1.text(5, -0.2, 'main', ha='center', fontsize=10, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1.5))
    ax1.text(7, 2.2, 'feature', ha='center', fontsize=10, fontweight='bold',
             color=SUCCESS, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f8f5', ec=SUCCESS, lw=1.5))

    # ── Bottom: Both branches have diverged ──
    ax2.set_xlim(-0.5, 10); ax2.set_ylim(-0.5, 2.8)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('Step 2: Commit on main too — branches diverge!', fontsize=13,
                  fontweight='bold', color=ACCENT, loc='left', pad=5)

    for x, lbl in [(1, 'c1'), (3, 'c2'), (5, 'c3')]:
        draw_c(ax2, x, 1.0, lbl, PRIMARY, r)
    draw_c(ax2, 7, 2.0, 'A', SUCCESS, r)
    draw_c(ax2, 7, 0.0, 'B', ACCENT, r)
    ax2.plot([1+r, 3-r], [1, 1], color=PRIMARY, lw=2.5)
    ax2.plot([3+r, 5-r], [1, 1], color=PRIMARY, lw=2.5)
    ax2.plot([5+0.15, 7-r], [1+0.15, 2.0], color=SUCCESS, lw=2.5)
    ax2.plot([5+0.15, 7-r], [1-0.15, 0.0], color=ACCENT, lw=2.5)

    ax2.text(7, -0.7, 'main', ha='center', fontsize=10, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1.5))
    ax2.text(7, 2.55, 'feature', ha='center', fontsize=10, fontweight='bold',
             color=SUCCESS, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f8f5', ec=SUCCESS, lw=1.5))
    ax2.text(9, 1.0, 'Now they\nhave diverged.\nNeed a merge!', ha='center', fontsize=10,
             color=ACCENT, style='italic', fontweight='bold')

    plt.tight_layout(h_pad=1.5)
    fig.savefig('git_diverging_branches.png', dpi=200, bbox_inches='tight',
                facecolor=WHITE, edgecolor='none')
    plt.close()
    print('  Created git_diverging_branches.png')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
# 15. REBASE: before and after
# ═══════════════════════════════════════════════════════════════════════════

def draw_rebase():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6))
    fig.patch.set_facecolor(WHITE)
    r = 0.25

    def draw_c(ax, x, y, label, color, radius=0.25, alpha=1.0):
        circle = plt.Circle((x, y), radius, color=color, ec='white',
                            lw=2, zorder=3, alpha=alpha)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=4, family='monospace',
                alpha=alpha)

    # ── Top: Before rebase ──
    ax1.set_xlim(-0.5, 11); ax1.set_ylim(-0.5, 2.8)
    ax1.set_aspect('equal'); ax1.axis('off')
    ax1.set_title('Before: git rebase main', fontsize=13, fontweight='bold',
                  color=PRIMARY, loc='left', pad=5)

    # Main: A-B-C
    for x, lbl in [(1, 'A'), (3, 'B'), (5, 'C')]:
        draw_c(ax1, x, 0.5, lbl, PRIMARY, r)
    ax1.plot([1+r, 3-r], [0.5, 0.5], color=PRIMARY, lw=2.5)
    ax1.plot([3+r, 5-r], [0.5, 0.5], color=PRIMARY, lw=2.5)
    ax1.text(5, -0.25, 'main', ha='center', fontsize=10, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1.5))

    # Feature: D-E branching from B
    draw_c(ax1, 5, 2.0, 'D', SUCCESS, r)
    draw_c(ax1, 7, 2.0, 'E', SUCCESS, r)
    ax1.plot([3+0.15, 5-r], [0.5+0.15, 2.0], color=SUCCESS, lw=2.5)
    ax1.plot([5+r, 7-r], [2.0, 2.0], color=SUCCESS, lw=2.5)
    ax1.text(7, 2.6, 'feature', ha='center', fontsize=10, fontweight='bold',
             color=SUCCESS, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f8f5', ec=SUCCESS, lw=1.5))

    # ── Bottom: After rebase ──
    ax2.set_xlim(-0.5, 11); ax2.set_ylim(-0.5, 2.0)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('After: commits replayed on top of main (new hashes!)', fontsize=13,
                  fontweight='bold', color=SUCCESS, loc='left', pad=5)

    # Main: A-B-C
    for x, lbl in [(1, 'A'), (3, 'B'), (5, 'C')]:
        draw_c(ax2, x, 0.5, lbl, PRIMARY, r)
    ax2.plot([1+r, 3-r], [0.5, 0.5], color=PRIMARY, lw=2.5)
    ax2.plot([3+r, 5-r], [0.5, 0.5], color=PRIMARY, lw=2.5)
    ax2.text(5, -0.25, 'main', ha='center', fontsize=10, fontweight='bold',
             color=PRIMARY, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f0fe', ec=PRIMARY, lw=1.5))

    # Rebased: D'-E' after C
    draw_c(ax2, 7, 0.5, "D'", SUCCESS, r)
    draw_c(ax2, 9, 0.5, "E'", SUCCESS, r)
    ax2.plot([5+r, 7-r], [0.5, 0.5], color=SUCCESS, lw=2.5)
    ax2.plot([7+r, 9-r], [0.5, 0.5], color=SUCCESS, lw=2.5)
    ax2.text(9, -0.25, 'feature', ha='center', fontsize=10, fontweight='bold',
             color=SUCCESS, bbox=dict(boxstyle='round,pad=0.15', fc='#e8f8f5', ec=SUCCESS, lw=1.5))

    ax2.text(10.3, 0.5, 'Linear\nhistory!', ha='center', fontsize=11,
             color=SUCCESS, style='italic', fontweight='bold')

    plt.tight_layout(h_pad=1.5)
    fig.savefig('git_rebase.png', dpi=200, bbox_inches='tight',
                facecolor=WHITE, edgecolor='none')
    plt.close()
    print('  Created git_rebase.png')


# ═══════════════════════════════════════════════════════════════════════════
# 16. GIT OBJECT MODEL (DAG view: commit → tree → blobs)
# ═══════════════════════════════════════════════════════════════════════════

def draw_object_dag():
    g = graphviz.Digraph('object_dag', format='png',
                         graph_attr={'rankdir': 'TB', 'bgcolor': WHITE,
                                     'dpi': '200', 'pad': '0.5',
                                     'nodesep': '0.5', 'ranksep': '0.7',
                                     'label': 'Git Object Model: Everything is a Hash',
                                     'labelloc': 't', 'fontsize': '16',
                                     'fontname': 'Helvetica Bold',
                                     'fontcolor': PRIMARY})

    # Commit
    g.node('commit', label='''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD><B>COMMIT</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">tree: a3f2...</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">parent: 7c1...</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">msg: "Add model"</FONT></TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded',
           fillcolor='#e8f4fd', color=PRIMARY_LIGHT, penwidth='2',
           fontname='Helvetica', fontsize='11')

    # Tree
    g.node('tree', label='''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD><B>TREE</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">train.py → 5a3...</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">utils.py → 8b7...</FONT></TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded',
           fillcolor='#fef9e7', color='#d4a017', penwidth='2',
           fontname='Helvetica', fontsize='11')

    # Blobs
    g.node('blob1', label='''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD><B>BLOB</B></TD></TR>
        <TR><TD><FONT FACE="Courier" POINT-SIZE="9">import numpy...</FONT></TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded',
           fillcolor='#e8f8f5', color=SUCCESS, penwidth='2',
           fontname='Helvetica', fontsize='11')

    g.node('blob2', label='''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD><B>BLOB</B></TD></TR>
        <TR><TD><FONT FACE="Courier" POINT-SIZE="9">def accuracy...</FONT></TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded',
           fillcolor='#e8f8f5', color=SUCCESS, penwidth='2',
           fontname='Helvetica', fontsize='11')

    # Parent commit
    g.node('parent', label='''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD><B>PARENT COMMIT</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">(previous snapshot)</FONT></TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded',
           fillcolor='#f3e8ff', color=PURPLE, penwidth='2',
           fontname='Helvetica', fontsize='11')

    # Formula
    g.node('formula', label='''<<TABLE BORDER="0" CELLBORDER="0">
        <TR><TD><FONT POINT-SIZE="10"><I>hash = SHA-1(type + size + content)</I></FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">Any change → new hash → new object</FONT></TD></TR>
    </TABLE>>''', shape='none')

    g.edge('commit', 'tree', label=' tree ', fontname='Helvetica Bold',
           fontsize='9', color=PRIMARY, penwidth='2')
    g.edge('commit', 'parent', label=' parent ', fontname='Helvetica Bold',
           fontsize='9', color=PURPLE, penwidth='2', style='dashed')
    g.edge('tree', 'blob1', label=' train.py ', fontname='Courier',
           fontsize='9', color=SUCCESS, penwidth='1.5')
    g.edge('tree', 'blob2', label=' utils.py ', fontname='Courier',
           fontsize='9', color=SUCCESS, penwidth='1.5')

    g.render('git_object_dag', cleanup=True)
    print('  Created git_object_dag.png')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating Week 08 Git diagrams...")
    draw_three_areas()          # 1
    draw_file_lifecycle()       # 2
    draw_branch_merge()         # 3
    draw_ff_vs_threeway()       # 4
    draw_feature_workflow()     # 5
    draw_reset_modes()          # 6
    draw_local_remote()         # 7
    draw_conflict_flow()        # 8
    draw_stash_workflow()       # 9
    draw_commit_anatomy()       # 10
    draw_snapshot_model()       # 11
    draw_commit_chain()         # 12
    draw_branch_pointer()       # 13
    draw_diverging_branches()   # 14
    draw_rebase()               # 15
    draw_object_dag()           # 16
    print(f"Done! Generated 16 diagrams.")

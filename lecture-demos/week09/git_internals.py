#!/usr/bin/env python3
"""
Git Internals Explorer — Week 09 Lecture Demo
=============================================
Run this script to see what ACTUALLY happens inside .git/
at every step. Each section pauses for you to read and explore.

Usage:
    python git_internals.py                  # Build a fresh demo repo
    python git_internals.py --step           # Pause between sections (lecture mode)
    python git_internals.py --repo /path     # Explore an existing repo's internals
    python git_internals.py --repo . --step  # Explore current dir, with pauses

Requires: Python 3.8+, git
"""

import argparse, subprocess, os, sys, tempfile, shutil, hashlib, zlib, textwrap
from pathlib import Path

# ─── CLI args ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Git Internals Explorer")
parser.add_argument("--step", action="store_true",
                    help="Pause between sections (lecture mode)")
parser.add_argument("--repo", type=str, default=None,
                    help="Path to an existing Git repo to explore")
args = parser.parse_args()

STEP_MODE = args.step
EXPLORE_REPO = args.repo

# ─── Colors ─────────────────────────────────────────────────────────────────

BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def section(title):
    print(f"\n{BLUE}{'━' * 60}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"{BLUE}{'━' * 60}{RESET}\n")

def explain(text):
    for line in textwrap.dedent(text).strip().split("\n"):
        print(f"  {DIM}{line}{RESET}")
    print()

def run(cmd, show=True, cwd=None):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True,
                            text=True, cwd=cwd or os.getcwd())
    output = result.stdout.strip()
    if show and output:
        for line in output.split("\n"):
            print(f"  {line}")
    if result.returncode != 0 and result.stderr.strip():
        for line in result.stderr.strip().split("\n"):
            if "hint:" not in line:
                print(f"  {RED}{line}{RESET}")
    return output

def show_cmd(cmd, cwd=None):
    """Show and run a command."""
    print(f"  {YELLOW}$ {cmd}{RESET}")
    out = run(cmd, cwd=cwd)
    print()
    return out

def pause():
    if STEP_MODE:
        input(f"  {GREEN}Press Enter to continue...{RESET}")
        print()

def show_file(path, label=None):
    """Show contents of a file with a label."""
    label = label or path
    print(f"  {CYAN}── {label} ──{RESET}")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                print(f"  {DIM}│{RESET} {line.rstrip()}")
    else:
        print(f"  {RED}(does not exist){RESET}")
    print()

def show_tree(directory, prefix="", max_depth=3, depth=0):
    """Show directory tree (like the `tree` command)."""
    if depth >= max_depth:
        return
    entries = sorted(Path(directory).iterdir())
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    for f in files:
        print(f"  {prefix}├── {f.name}")
    for d in dirs:
        print(f"  {prefix}├── {CYAN}{d.name}/{RESET}")
        show_tree(d, prefix + "│   ", max_depth, depth + 1)


# ═══════════════════════════════════════════════════════════════════════════
# EXPLORE MODE: inspect an existing repo
# ═══════════════════════════════════════════════════════════════════════════

def explore_existing_repo(repo_path):
    """Explore the internals of an existing Git repository."""
    repo_path = os.path.abspath(repo_path)

    # Find the git dir (could be a subdirectory that's in a repo)
    result = subprocess.run("git rev-parse --show-toplevel", shell=True,
                            capture_output=True, text=True, cwd=repo_path)
    if result.returncode != 0:
        print(f"  {RED}Error: '{repo_path}' is not inside a Git repository.{RESET}")
        sys.exit(1)
    repo_root = result.stdout.strip()
    os.chdir(repo_root)

    print(f"\n{BOLD}Git Internals Explorer — Existing Repo Mode{RESET}")
    print(f"Repo: {CYAN}{repo_root}{RESET}\n")

    # ── Section 1: .git/ structure ──────────────────────────────────────
    section("1. The .git/ directory")

    explain("""
        This is the hidden folder that IS your repository.
        Everything Git knows lives here:
    """)

    print(f"  {CYAN}.git/ top-level structure:{RESET}")
    show_tree(".git", max_depth=1)
    print()

    print(f"  {CYAN}HEAD (which branch are you on?):{RESET}")
    show_file(".git/HEAD", "HEAD")

    pause()

    # ── Section 2: Branches ─────────────────────────────────────────────
    section("2. All branches (just text files!)")

    explain("""
        Each branch is a file in .git/refs/heads/ containing a
        40-character SHA-1 hash pointing to a commit.
    """)

    branches = run("git branch -a", show=False).strip().split("\n")
    current = None
    for b in branches:
        b = b.strip()
        marker = ""
        if b.startswith("* "):
            b = b[2:]
            current = b
            marker = f" {GREEN}← current{RESET}"
        sha = run(f"git rev-parse {b} 2>/dev/null", show=False)
        if sha:
            print(f"  {YELLOW}{b:40s}{RESET} → {sha[:12]}...{marker}")
    print()

    pause()

    # ── Section 3: Recent commits ───────────────────────────────────────
    section("3. Recent commit objects")

    explain("""
        Let's look at the actual commit objects — what's inside them:
    """)

    commits = run("git log --oneline -5", show=False).strip().split("\n")
    for line in commits:
        sha = line.split()[0]
        print(f"  {CYAN}── Commit {sha} ──{RESET}")
        show_cmd(f"git cat-file -p {sha}")

    pause()

    # ── Section 4: Latest tree ──────────────────────────────────────────
    section("4. The tree object (directory snapshot)")

    explain("""
        The tree is Git's snapshot of your project's file structure.
        It maps filenames to blob hashes:
    """)

    tree_sha = run("git rev-parse HEAD^{tree}", show=False)
    print(f"  {CYAN}Tree {tree_sha[:12]}... (HEAD):{RESET}")
    show_cmd(f"git cat-file -p {tree_sha[:8]}")

    # Show a subdirectory tree if one exists
    tree_entries = run(f"git cat-file -p {tree_sha[:8]}", show=False)
    subtrees = [line for line in tree_entries.split("\n")
                if line.startswith("040000")]
    if subtrees:
        first_subtree = subtrees[0].split()
        sub_sha = first_subtree[2]
        sub_name = first_subtree[3]
        print(f"  {CYAN}Subtree '{sub_name}/' ({sub_sha[:12]}...):{RESET}")
        show_cmd(f"git cat-file -p {sub_sha[:8]}")

    pause()

    # ── Section 5: A blob (file contents) ───────────────────────────────
    section("5. Blob objects (file contents)")

    explain("""
        Blobs store raw file contents — no filename, no metadata.
        The filename is stored in the tree, not the blob.
        Let's peek at a few:
    """)

    # Find first few blob entries from the tree
    tree_lines = run(f"git cat-file -p {tree_sha[:8]}", show=False).strip().split("\n")
    blob_count = 0
    for line in tree_lines:
        parts = line.split()
        if len(parts) >= 4 and parts[1] == "blob":
            blob_sha = parts[2]
            blob_name = parts[3]
            size = run(f"git cat-file -s {blob_sha[:8]}", show=False)
            print(f"  {CYAN}── {blob_name} (blob {blob_sha[:12]}..., {size} bytes) ──{RESET}")
            content = run(f"git cat-file -p {blob_sha[:8]}", show=False)
            # Show first 10 lines only
            lines = content.split("\n")
            for l in lines[:10]:
                print(f"  {DIM}│{RESET} {l}")
            if len(lines) > 10:
                print(f"  {DIM}│ ... ({len(lines) - 10} more lines){RESET}")
            print()
            blob_count += 1
            if blob_count >= 3:
                break

    pause()

    # ── Section 6: Object statistics ────────────────────────────────────
    section("6. Object database statistics")

    explain("""
        Let's count ALL objects in the repository and categorize them.
        Note: packed objects (in .git/objects/pack/) are counted via
        git's internal tools, not just file listing.
    """)

    # Use git count-objects for a quick summary
    show_cmd("git count-objects -v")

    # Count by type using rev-list
    commit_count = run("git rev-list --all --count", show=False)
    print(f"  {CYAN}Object counts:{RESET}")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  commits    {commit_count:>6s}  (via rev-list --all)  │")

    # Count trees and blobs from the last commit's tree
    tree_count = run("git rev-list --all --objects | wc -l", show=False).strip()
    print(f"  │  all objects {tree_count:>5s}  (commits+trees+blobs) │")
    print(f"  └─────────────────────────────────────────┘")
    print()

    pause()

    # ── Section 7: Verify integrity ─────────────────────────────────────
    section("7. Integrity check")

    explain("""
        Every object's filename IS its SHA-1 hash.
        git fsck verifies that nothing is corrupted:
    """)

    show_cmd("git fsck --no-dangling 2>&1 | head -5")

    explain("""
        If any object's content doesn't match its hash,
        Git will report it here. Content-addressable storage
        with built-in integrity checking!
    """)

    pause()

    # ── Section 8: Commit graph ─────────────────────────────────────────
    section("8. The commit graph")

    show_cmd("git log --oneline --graph --all --decorate -20")

    # ── Done ────────────────────────────────────────────────────────────
    section("Exploration complete!")

    print(f"  {BOLD}Repo:{RESET} {repo_root}")
    print()
    print(f"  {CYAN}Commands to keep exploring:{RESET}")
    print(f"  git cat-file -p HEAD            # See latest commit")
    print(f"  git cat-file -p HEAD^{{tree}}     # See the file tree")
    print(f"  git cat-file -t <sha>           # Check object type")
    print(f"  git cat-file -p <sha>           # Print object contents")
    print(f"  git rev-list --all --objects     # List all object SHAs")
    print(f"  git log --oneline --graph --all  # See the full graph")
    print()
    print(f"  {BOLD}{GREEN}Happy exploring!{RESET}")


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCH: --repo mode or demo mode
# ═══════════════════════════════════════════════════════════════════════════

if EXPLORE_REPO:
    explore_existing_repo(EXPLORE_REPO)
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════════════════
# DEMO MODE SETUP
# ═══════════════════════════════════════════════════════════════════════════

DEMO_DIR = tempfile.mkdtemp(prefix="git-internals-")
REPO_DIR = os.path.join(DEMO_DIR, "ml-project")
print(f"\n{BOLD}Git Internals Explorer{RESET}")
print(f"Working in: {CYAN}{DEMO_DIR}{RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 1. WHAT DOES `git init` ACTUALLY CREATE?
# ═══════════════════════════════════════════════════════════════════════════

section("1. What does `git init` actually create?")

os.makedirs(REPO_DIR)
os.chdir(REPO_DIR)
run("git init", show=False)
run("git checkout -b main 2>/dev/null", show=False)
run("git config user.name 'Demo Student'", show=False)
run("git config user.email 'student@iitgn.ac.in'", show=False)

explain("""
    When you run `git init`, Git creates a hidden .git/ folder.
    Let's look inside it — this IS your repository:
""")

print(f"  {CYAN}.git/ directory structure:{RESET}")
show_tree(".git", max_depth=2)

explain("""
    Key folders:
    • objects/  — Where Git stores ALL data (files, commits, trees)
    • refs/     — Branch pointers (just files containing SHA hashes)
    • HEAD      — Points to the current branch
""")

print(f"  {CYAN}What does HEAD contain?{RESET}")
show_file(".git/HEAD", "HEAD")

explain("""
    HEAD says "ref: refs/heads/main" — meaning "I'm on the main branch."
    But refs/heads/main doesn't exist yet (no commits!).
""")

print(f"  {CYAN}How many objects exist?{RESET}")
count = show_cmd("find .git/objects -type f | wc -l")
explain("    Zero objects! The repository is empty.\n")

pause()


# ═══════════════════════════════════════════════════════════════════════════
# 2. WHAT DOES `git add` ACTUALLY DO?
# ═══════════════════════════════════════════════════════════════════════════

section("2. What does `git add` actually do?")

# Create a file
with open("train.py", "w") as f:
    f.write('import numpy as np\n\ndef train(X, y):\n    return np.mean(y)\n')

explain("""
    We just created train.py. Let's see what Git knows:
""")

show_cmd("git status --short")

explain("    ?? means untracked — Git sees the file but isn't tracking it.\n")

# Compute the hash ourselves!
with open("train.py", "rb") as f:
    content = f.read()

# Git blob format: "blob <size>\0<content>"
header = f"blob {len(content)}\0".encode()
blob_data = header + content
sha1 = hashlib.sha1(blob_data).hexdigest()

print(f"  {CYAN}Let's compute the SHA-1 hash ourselves:{RESET}")
print(f"  Git stores files as: blob <size>\\0<content>")
print(f"  SHA-1 of that = {GREEN}{sha1}{RESET}")
print()

# Now actually git add
show_cmd("git add train.py")

explain("    Now let's see what changed inside .git/:\n")

show_cmd("find .git/objects -type f")

explain(f"""
    A new object appeared! Its path is:
    .git/objects/{sha1[:2]}/{sha1[2:]}

    The SHA-1 hash we computed matches! Git uses the first 2 chars
    as a directory name and the rest as the filename.
""")

# Show object type and content
print(f"  {CYAN}What type of object is it?{RESET}")
show_cmd(f"git cat-file -t {sha1[:8]}")

print(f"  {CYAN}What's inside it?{RESET}")
show_cmd(f"git cat-file -p {sha1[:8]}")

explain("""
    It's a "blob" — Git's name for a file's contents.
    A blob is just the raw file content, nothing more.
    No filename! The name is stored elsewhere (in trees).
""")

# Show the staging area (index)
print(f"  {CYAN}The staging area is stored in .git/index:{RESET}")
show_cmd("git ls-files --stage")

explain("""
    The staging area (index) maps filenames to blob hashes.
    It says: "train.py should contain blob <sha1>"
""")

pause()


# ═══════════════════════════════════════════════════════════════════════════
# 3. WHAT DOES `git commit` ACTUALLY DO?
# ═══════════════════════════════════════════════════════════════════════════

section("3. What does `git commit` actually do?")

explain("""
    A commit needs THREE things:
    1. A "tree" object — snapshot of the file structure
    2. A parent pointer — the previous commit (none for first commit)
    3. Metadata — author, date, message
""")

run("git commit -m 'Add training script'", show=False)
print(f"  {GREEN}✓ First commit created{RESET}\n")

show_cmd("find .git/objects -type f | wc -l")

explain("    We went from 1 object (blob) to 3 objects! Let's see them:\n")

# List all objects and their types
objects = run("find .git/objects -type f", show=False).strip().split("\n")
print(f"  {CYAN}All objects and their types:{RESET}")
for obj_path in objects:
    sha = obj_path.replace(".git/objects/", "").replace("/", "")
    obj_type = run(f"git cat-file -t {sha}", show=False)
    obj_size = run(f"git cat-file -s {sha}", show=False)
    print(f"  {sha[:12]}...  {YELLOW}{obj_type:6s}{RESET}  ({obj_size} bytes)")
print()

# Show the tree object
commit_sha = run("git rev-parse HEAD", show=False)
tree_sha = run(f"git rev-parse HEAD^{{tree}}", show=False)

print(f"  {CYAN}The commit object:{RESET}")
show_cmd(f"git cat-file -p {commit_sha[:8]}")

print(f"  {CYAN}The tree object (snapshot of files):{RESET}")
show_cmd(f"git cat-file -p {tree_sha[:8]}")

explain("""
    The TREE maps filenames to blobs:
    • "train.py" → blob <sha1>

    The COMMIT points to the tree and adds metadata:
    • tree → <tree-sha>
    • author, committer, message

    This is the complete data model:
      COMMIT → TREE → BLOB(s)
""")

# Show that branch pointer is now set
print(f"  {CYAN}The branch pointer (refs/heads/main):{RESET}")
show_file(".git/refs/heads/main", "refs/heads/main")

explain(f"""
    The branch "main" is just a file containing the commit hash!
    That's why branching in Git is instant — it just writes a
    40-character string to a file.
""")

pause()


# ═══════════════════════════════════════════════════════════════════════════
# 4. SECOND COMMIT — SEE THE CHAIN FORM
# ═══════════════════════════════════════════════════════════════════════════

section("4. Second commit — watch the chain form")

first_commit = run("git rev-parse HEAD", show=False)

# Add a new file
with open("utils.py", "w") as f:
    f.write('def accuracy(y_true, y_pred):\n    return (y_true == y_pred).mean()\n')

run("git add utils.py && git commit -m 'Add accuracy function'", show=False)
print(f"  {GREEN}✓ Second commit created{RESET}\n")

second_commit = run("git rev-parse HEAD", show=False)

show_cmd("find .git/objects -type f | wc -l")

explain("    Now we have 6 objects! Let's see the new commit:\n")

print(f"  {CYAN}Second commit:{RESET}")
show_cmd(f"git cat-file -p {second_commit[:8]}")

print(f"  {CYAN}Notice the 'parent' line!{RESET}")
explain(f"""
    The second commit has:
    • tree → new snapshot (tree with both train.py AND utils.py)
    • parent → {first_commit[:12]}... (the first commit!)

    This parent pointer is how Git creates the history chain:
      commit2 → commit1 → (none, root)
""")

# Show the new tree has both files
tree2 = run(f"git rev-parse HEAD^{{tree}}", show=False)
print(f"  {CYAN}New tree (snapshot of ALL files):{RESET}")
show_cmd(f"git cat-file -p {tree2[:8]}")

explain("""
    The new tree has TWO entries: train.py and utils.py.
    Note: train.py's blob hash is the SAME as before — Git
    doesn't duplicate unchanged files!
""")

pause()


# ═══════════════════════════════════════════════════════════════════════════
# 5. BRANCHES ARE JUST POINTERS
# ═══════════════════════════════════════════════════════════════════════════

section("5. Branches are just files with SHA hashes")

explain("""
    Let's prove that a branch is literally just a text file:
""")

print(f"  {CYAN}Before creating a branch:{RESET}")
show_cmd("ls .git/refs/heads/")

run("git checkout -b feature/augmentation 2>/dev/null", show=False)
print(f"  {GREEN}✓ Created branch 'feature/augmentation'{RESET}\n")

print(f"  {CYAN}After creating a branch:{RESET}")
show_cmd("ls .git/refs/heads/")

# Show that both files contain the SAME hash
print(f"  {CYAN}Contents of both branch files:{RESET}")
show_cmd("cat .git/refs/heads/main")
show_cmd("cat .git/refs/heads/feature/augmentation")

explain("""
    Both branches point to the SAME commit! Creating a branch
    just created a new file. No copying, no overhead.
""")

print(f"  {CYAN}HEAD now points to the new branch:{RESET}")
show_file(".git/HEAD", "HEAD")

explain("""
    HEAD changed from "ref: refs/heads/main" to
    "ref: refs/heads/feature/augmentation".
    That's ALL that `git checkout -b` did:
    1. Create a new file in refs/heads/
    2. Update HEAD to point to it
""")

pause()


# ═══════════════════════════════════════════════════════════════════════════
# 6. COMMITS ON A BRANCH — WATCH POINTERS DIVERGE
# ═══════════════════════════════════════════════════════════════════════════

section("6. Commits on a branch — pointers diverge")

with open("augment.py", "w") as f:
    f.write('import numpy as np\n\ndef add_noise(X, scale=0.1):\n    return X + np.random.normal(0, scale, X.shape)\n')

run("git add augment.py && git commit -m 'Add augmentation module'", show=False)
print(f"  {GREEN}✓ Committed on feature/augmentation{RESET}\n")

feature_sha = run("git rev-parse feature/augmentation", show=False)
main_sha = run("git rev-parse main", show=False)

print(f"  {CYAN}Branch pointers now:{RESET}")
print(f"  main                   → {GREEN}{main_sha[:12]}...{RESET}")
print(f"  feature/augmentation   → {GREEN}{feature_sha[:12]}...{RESET}")
print()

explain(f"""
    They point to DIFFERENT commits now!
    • main is still at the second commit
    • feature/augmentation moved forward to the third commit
""")

print(f"  {CYAN}The graph:{RESET}")
show_cmd("git log --oneline --graph --all --decorate")

pause()


# ═══════════════════════════════════════════════════════════════════════════
# 7. MERGE — HOW GIT COMBINES TREES
# ═══════════════════════════════════════════════════════════════════════════

section("7. Merge — how Git combines trees")

run("git checkout main 2>/dev/null", show=False)

explain("""
    Let's merge feature/augmentation into main.
    Since main hasn't changed, this will be a fast-forward:
""")

before_main = run("git rev-parse main", show=False)
show_cmd("git merge feature/augmentation")

after_main = run("git rev-parse main", show=False)

print(f"  {CYAN}What changed?{RESET}")
print(f"  main before: {before_main[:12]}...")
print(f"  main after:  {after_main[:12]}...")
print()

explain("""
    Fast-forward: Git just moved the main pointer forward.
    No new objects created! Let's verify:
""")

show_cmd("cat .git/refs/heads/main")
show_cmd("cat .git/refs/heads/feature/augmentation")

explain("""
    Both files now contain the same hash. That's all a
    fast-forward merge does — update a pointer.
""")

pause()

# Now create a real three-way merge
run("git checkout -b feature/evaluation 2>/dev/null", show=False)
with open("evaluate.py", "w") as f:
    f.write('def evaluate(model, X, y):\n    return model.score(X, y)\n')
run("git add evaluate.py && git commit -m 'Add evaluation module'", show=False)

run("git checkout main 2>/dev/null", show=False)
with open("README.md", "w") as f:
    f.write('# ML Project\nA simple ML pipeline.\n')
run("git add README.md && git commit -m 'Add README'", show=False)

explain("    Now main AND feature/evaluation have diverged. Let's merge:\n")

obj_before = int(run("find .git/objects -type f | wc -l", show=False))
show_cmd("git merge feature/evaluation -m 'Merge evaluation module'")

obj_after = int(run("find .git/objects -type f | wc -l", show=False))

print(f"  {CYAN}Objects before merge: {obj_before}{RESET}")
print(f"  {CYAN}Objects after merge:  {obj_after}{RESET}")
print(f"  {CYAN}New objects created:  {obj_after - obj_before}{RESET}\n")

explain("""
    A three-way merge creates NEW objects:
    • A merge commit (with TWO parent pointers)
    • A new tree (combining files from both branches)
""")

merge_sha = run("git rev-parse HEAD", show=False)
print(f"  {CYAN}The merge commit has TWO parents:{RESET}")
show_cmd(f"git cat-file -p {merge_sha[:8]}")

show_cmd("git log --oneline --graph --all --decorate")

pause()


# ═══════════════════════════════════════════════════════════════════════════
# 8. THE OBJECT MODEL — A SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

section("8. The complete object model")

total = run("find .git/objects -type f | wc -l", show=False)

explain(f"""
    Our tiny repo has {total} objects. Let's categorize them all:
""")

objects = run("find .git/objects -type f", show=False).strip().split("\n")
counts = {"blob": 0, "tree": 0, "commit": 0}
for obj_path in objects:
    sha = obj_path.replace(".git/objects/", "").replace("/", "")
    obj_type = run(f"git cat-file -t {sha}", show=False)
    if obj_type in counts:
        counts[obj_type] += 1

print(f"  {CYAN}Object breakdown:{RESET}")
print(f"  ┌─────────────────────────────────────────┐")
print(f"  │  {'blob':8s}  {counts['blob']:3d}  (file contents)          │")
print(f"  │  {'tree':8s}  {counts['tree']:3d}  (directory snapshots)    │")
print(f"  │  {'commit':8s}  {counts['commit']:3d}  (snapshots + metadata)  │")
print(f"  │  {'TOTAL':8s}  {sum(counts.values()):3d}                          │")
print(f"  └─────────────────────────────────────────┘")
print()

explain("""
    Git's entire data model is just THREE types of objects:

    BLOB   — Raw file contents (no filename!)
    TREE   — Maps filenames → blobs (like a directory listing)
    COMMIT — Points to a tree + parent + metadata

    Everything else (branches, tags, HEAD) is just a pointer
    to one of these objects. That's the whole thing!

    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  COMMIT  │────▶│   TREE   │────▶│   BLOB   │
    │          │     │          │     │ (file.py) │
    │ message  │     │ file→blob│     │          │
    │ parent   │     │ file→blob│     └──────────┘
    └──────────┘     └──────────┘
         │
         ▼
    ┌──────────┐
    │  PARENT  │
    │ (commit) │
    └──────────┘
""")

pause()


# ═══════════════════════════════════════════════════════════════════════════
# 9. BONUS: VERIFY GIT'S INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

section("9. Bonus: Git's integrity guarantee")

explain("""
    Every object's filename IS its SHA-1 hash. This means:
    • If a single byte changes, the hash changes → corruption detected
    • Two identical files always get the same hash → automatic dedup
    • You can verify the entire repo with one command:
""")

show_cmd("git fsck")

explain("""
    `git fsck` checks that every object's hash matches its content.
    If any object is corrupted, Git will tell you immediately.

    This is why Git is so reliable — it's content-addressable storage
    with built-in integrity checking.
""")


# ═══════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════

section("Demo complete!")

print(f"  {BOLD}Repo location:{RESET} {REPO_DIR}")
print()
print(f"  {CYAN}Key takeaways:{RESET}")
print(f"  1. .git/ is just a folder with files")
print(f"  2. Objects are blobs (files), trees (directories), commits")
print(f"  3. Branches are text files containing a SHA-1 hash")
print(f"  4. HEAD is a text file pointing to the current branch")
print(f"  5. Everything is content-addressed (hash = filename)")
print()
print(f"  {CYAN}Explore on your own:{RESET}")
print(f"  cd {REPO_DIR}")
print(f"  git cat-file -p HEAD          # See latest commit")
print(f"  git cat-file -p HEAD^{{tree}}   # See the file tree")
print(f"  git cat-file -p <blob-sha>    # See a file's contents")
print(f"  find .git/objects -type f     # See all objects")
print()
print(f"  {BOLD}{GREEN}Happy exploring!{RESET}")

#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Git Deep Dive — Follow-Along Guide                                     ║
# ║  Week 8 · CS 203 · Software Tools and Techniques for AI                ║
# ║  Prof. Nipun Batra · IIT Gandhinagar                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# THE STORY (~80 minutes):
#   You're building an ML project. You start the way everyone does —
#   copying files, renaming them _v2_FINAL, losing track. Then you
#   discover Git and never look back. By the end, you're collaborating
#   with teammates through a shared remote.
#
# HOW TO USE:
#   1. Open this file in your editor (VS Code, etc.)
#   2. Open a terminal side-by-side
#   3. Copy-paste each command into your terminal, one at a time
#   4. Compare your output with the expected output shown here
#   5. DO NOT run this file as a script — read it and type along
#
# LEGEND:
#   Lines without # prefix     →  commands to type
#   # >> ...                   →  expected output
#   # ...                      →  explanation / narration
#   # [SLIDE: ...]             →  instructor: show this slide
#
# ═══════════════════════════════════════════════════════════════════════════



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 2: "The Mess" (version_control_chaos.png)        ║
# ║     Show the chaos image. Then say "Let's feel the pain ourselves."     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 1: Life Without Git (The Mess)                          ~5 min     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# You're starting an ML project. No version control. Just files and hope.

mkdir ml-chaos && cd ml-chaos

# Write your first training script:

cat > train.py << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression

def train(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

print("Training model...")
EOF

# It works! You show your advisor. They want changes.
# But you're scared to break what works. So you make a copy:

cp train.py train_v2.py

# Edit train_v2.py to add evaluation:

cat > train_v2.py << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression

def train(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def evaluate(model, X, y):
    acc = model.score(X, y)
    print(f"Accuracy: {acc:.3f}")

print("Training and evaluating model...")
EOF

# Advisor likes it, wants more changes. You're afraid again:

cp train_v2.py train_v2_FINAL.py

# Oops, bug. Fix it:

cp train_v2_FINAL.py train_v2_FINAL_fixed.py

# New idea:

cp train_v2_FINAL_fixed.py train_v2_FINAL_fixed_actually_final.py

# Look at what we've created:

ls *.py

# Expected output:
# >> train.py
# >> train_v2.py
# >> train_v2_FINAL.py
# >> train_v2_FINAL_fixed.py
# >> train_v2_FINAL_fixed_actually_final.py

# Five copies. Now try to answer these questions:
#
#   Q1: Which file is the "real" current version?
#   Q2: What exactly changed between v2 and v2_FINAL?
#   Q3: Can I safely delete train.py?
#   Q4: My advisor says "go back to what we had Tuesday." Which file?
#
# Let's try Q2 — what changed between v2 and FINAL?

diff train_v2.py train_v2_FINAL.py

# Expected output: NOTHING. They're identical copies!
# You copied instead of editing. Now you can't tell them apart.

# Now imagine a teammate emails you THEIR version:

cat > train_alice.py << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression

def train(X, y):
    model = LogisticRegression(max_iter=500)   # Alice changed this
    model.fit(X, y)
    return model

def evaluate(model, X, y):
    acc = model.score(X, y)
    print(f"Accuracy: {acc:.3f}")

def preprocess(X):                              # Alice added this
    return (X - X.mean()) / X.std()

print("Alice's version")
EOF

# Now you have SIX files. Who has the right max_iter?
# Is Alice's preprocess() function useful? How do you combine her
# work with yours without losing anything?

ls *.py | wc -l

# >> 6

# This is unsustainable. Let's fix this properly.

cd ..
rm -rf ml-chaos



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 3–4: Letter Analogy + Three Areas               ║
# ║     Show both diagrams. Explain: edit → add → commit = write → envelope ║
# ║     → mailbox. Then say "Let's try it."                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 2: Starting Fresh With Git                              ~8 min     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Same project. This time, Git tracks everything.

mkdir ml-project && cd ml-project
git init

# Expected output:
# >> Initialized empty Git repository in /path/to/ml-project/.git/

# What just happened? Let's look:

ls -la

# >> .git/     <-- hidden folder. This IS the repository.

# Everything Git knows lives in .git/. Your project files live
# alongside it in the "working directory." Delete .git/ and it's
# just a normal folder again.

# Tell Git who you are (one-time setup):

git config user.name "Your Name"
git config user.email "your.email@example.com"

# TIP: Add --global to set this for ALL repos on your machine:
#   git config --global user.name "Your Name"

# Now let's recreate our training script — the SAME first version:

cat > train.py << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression

def train(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

print("Training model...")
EOF

# Ask Git what it sees:

git status

# Expected output:
# >> On branch main
# >> No commits yet
# >> Untracked files:
# >>   (use "git add <file>..." to include in what will be committed)
# >>         train.py
# >> nothing added to commit but untracked files present

# Key things to notice:
#  1. Git tells you WHAT to do next ("use git add...")
#  2. train.py is in RED — it's "untracked" (Git sees it, doesn't manage it)
#
# ╔══════════════════════════════════════════════════════════╗
# ║  Git has THREE areas:                                    ║
# ║                                                          ║
# ║  Working Directory  →  Staging Area  →  Repository       ║
# ║    (your files)       ("the envelope")   (saved history) ║
# ║                                                          ║
# ║  You edit here.       git add puts      git commit       ║
# ║                       things here.      saves them       ║
# ║                                         permanently.     ║
# ║                                                          ║
# ║  Think of it like writing a letter:                      ║
# ║    Write it → put in envelope → drop in mailbox          ║
# ╚══════════════════════════════════════════════════════════╝
#
# Right now train.py is only in the Working Directory.
# Let's move it to the Staging Area:

git add train.py

# What changed?

git status

# Expected output:
# >> Changes to be committed:
# >>   (use "git rm --cached <file>..." to unstage)
# >>         new file:   train.py

# Now train.py is GREEN — it's staged.
# It's in the "envelope," ready to be saved permanently.
# But it's NOT committed yet. One more step:

git commit -m "Add initial training script"

# Expected output:
# >> [main (root-commit) abc1234] Add initial training script
# >>  1 file changed, 10 insertions(+)
# >>  create mode 100644 train.py

# Let's see our history:

git log

# Expected output:
# >> commit abc1234... (HEAD -> main)
# >> Author: Your Name <your.email@example.com>
# >> Date:   Mon Mar 3 10:00:00 2026 +0530
# >>
# >>     Add initial training script

# That's verbose. Compact version:

git log --oneline

# >> abc1234 (HEAD -> main) Add initial training script

# ONE commit. One checkpoint. No copies. No _v2.
# The message explains WHY this checkpoint exists.

# Let's prove that the staging area matters. Create TWO files:

cat > utils.py << 'EOF'
import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
EOF

cat > notes.txt << 'EOF'
TODO: ask advisor about learning rate
meeting notes from Monday
not ready to commit yet
EOF

git status

# Expected output:
# >>   utils.py     (RED — untracked)
# >>   notes.txt    (RED — untracked)

# We want to commit utils.py but NOT notes.txt (it's not ready).
# The staging area lets us choose:

git add utils.py

git status

# Expected output:
# >> Changes to be committed:
# >>         new file:   utils.py       (GREEN — staged)
# >> Untracked files:
# >>         notes.txt                  (RED — not staged)

# Only utils.py will be in this commit. notes.txt stays out.

git commit -m "Add utils module with accuracy function"

git status

# Expected output:
# >> Untracked files:
# >>         notes.txt

# notes.txt is still there, still untracked. The staging area
# gave us fine-grained control over what goes into each commit.

# Clean up the notes file (we'll add it later):
rm notes.txt



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 5: "Acts 3–4 — Changes & Time Travel"            ║
# ║     Brief recap slide. Then stay in terminal for Acts 3 and 4.          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 3: Making Changes (No More Copies!)                    ~10 min     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Your advisor says: "Add an evaluation function."
# Old you would copy train.py → train_v2.py.
# New you just... edits the file.

cat > train.py << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression

def train(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def evaluate(model, X, y):
    acc = model.score(X, y)
    print(f"Accuracy: {acc:.3f}")
    return acc

print("Training and evaluating model...")
EOF

# "But what if I break something?" you worry.
# Relax. Let's see exactly what changed:

git status

# Expected output:
# >>   modified:   train.py    (RED — changed but not staged)

# It says "modified" not "untracked" — Git already knows this file.
# Let's see the EXACT changes, line by line:

git diff

# Expected output:
# >> diff --git a/train.py b/train.py
# >> --- a/train.py
# >> +++ b/train.py
# >> @@ -6,4 +6,10 @@ def train(X, y):
# >>      model.fit(X, y)
# >>      return model
# >>
# >> +def evaluate(model, X, y):
# >> +    acc = model.score(X, y)
# >> +    print(f"Accuracy: {acc:.3f}")
# >> +    return acc
# >> +
# >> -print("Training model...")
# >> +print("Training and evaluating model...")

# How to read this:
#   --- a/train.py   = the OLD version (last commit)
#   +++ b/train.py   = the NEW version (your working directory)
#   Lines with +     = added (shown in green)
#   Lines with -     = removed (shown in red)
#   Lines with space = unchanged (context)
#
# You can see EXACTLY what's different. No more guessing.

git add train.py
git commit -m "Add evaluation function"

# Let's add a config file:

cat > config.yaml << 'EOF'
model:
  type: logistic_regression
  max_iter: 1000
  learning_rate: 0.01

data:
  train_path: data/train.npy
  test_path: data/test.npy
EOF

git add config.yaml
git commit -m "Add model config file"

# Now update utils.py with more functions:

cat > utils.py << 'EOF'
import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def normalize(X):
    """Zero-mean, unit-variance normalization."""
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
EOF

git diff

# Shows the new confusion_matrix and normalize functions highlighted.

git add utils.py
git commit -m "Add confusion matrix and normalization to utils"

# And a preprocessing script:

cat > preprocess.py << 'EOF'
import numpy as np
from utils import normalize

def load_and_preprocess(path):
    """Load data and apply normalization."""
    data = np.load(path)
    X, y = data[:, :-1], data[:, -1]
    X = normalize(X)
    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess("data/train.npy")
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
EOF

git add preprocess.py
git commit -m "Add preprocessing pipeline with normalization"

# Let's see our full history:

git log --oneline

# Expected output:
# >> f6f7g8h (HEAD -> main) Add preprocessing pipeline with normalization
# >> e5e6f7g Add confusion matrix and normalization to utils
# >> d4d5e6f Add model config file
# >> c3c4d5e Add evaluation function
# >> b2b3c4d Add utils module with accuracy function
# >> a1a2b3c Add initial training script

# SIX checkpoints. No copies. Each one explains what changed and why.

# Let's also see which files changed in each commit:

git log --oneline --stat

# This shows file change summaries alongside each commit message.
# Very useful for getting the big picture.

# Now let's see how multiple file changes work in one commit.
# Update train.py to use the preprocessing:

cat > train.py << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression
from preprocess import load_and_preprocess
from utils import accuracy

def train(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def evaluate(model, X, y):
    preds = model.predict(X)
    acc = accuracy(y, preds)
    print(f"Accuracy: {acc:.3f}")
    return acc

if __name__ == "__main__":
    X, y = load_and_preprocess("data/train.npy")
    model = train(X, y)
    evaluate(model, X, y)
EOF

git status

# >> modified: train.py

# What if we also want to change config.yaml at the same time?

cat > config.yaml << 'EOF'
model:
  type: logistic_regression
  max_iter: 1000
  learning_rate: 0.01
  normalize: true

data:
  train_path: data/train.npy
  test_path: data/test.npy
  validation_split: 0.2
EOF

git status

# Expected output:
# >>   modified:   config.yaml
# >>   modified:   train.py

# Two files changed. Let's see what changed in EACH:

git diff train.py
git diff config.yaml

# You can also see everything at once:

git diff

# Stage both and commit:

git add train.py config.yaml
git commit -m "Integrate preprocessing into training pipeline"

# Before committing, there's a useful trick:
# What if you staged a file and want to see what's staged vs what's not?

echo "# TODO: add validation" >> train.py

git add train.py

echo "# TODO: add early stopping" >> train.py

# Now train.py has BOTH staged AND unstaged changes!

git status

# Expected output:
# >>   modified:   train.py    (GREEN — staged change: validation TODO)
# >>   modified:   train.py    (RED — unstaged change: early stopping TODO)

# See staged changes (what would go into the commit):
git diff --staged

# See unstaged changes (what's NOT staged yet):
git diff

# Let's clean this up — we don't actually want these TODOs:

git restore --staged train.py
git restore train.py

# ╔══════════════════════════════════════════════════════════╗
# ║  THE RHYTHM: edit → status → diff → add → commit        ║
# ║                                                          ║
# ║  git status      =  "What changed?"                     ║
# ║  git diff        =  "What exactly changed?"             ║
# ║  git diff --staged = "What am I about to commit?"       ║
# ║  git add <files> =  "Include these in the next commit"  ║
# ║  git commit -m   =  "Save this checkpoint"              ║
# ╚══════════════════════════════════════════════════════════╝



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Stay on Slide 5 (or hide slides — terminal only)       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 4: Going Back in Time                                  ~8 min     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Your advisor says: "The preprocessing is wrong. What did the code
# look like before you added it?"
#
# Old you: "Uh... let me check train_v2_FINAL_fixed... or was it v2?"
# New you:

git log --oneline

# See every checkpoint. Pick any one. Let's explore.


# --- INSPECTING COMMITS ---

# See what a specific commit changed (full diff):

git show HEAD~1

# Shows the "Integrate preprocessing" commit — full diff of what changed.

# See just the files that were touched:

git show --stat HEAD~1

# See what train.py looked like at the very first commit:

git show HEAD~6:train.py

# Expected output: the original train.py, exactly as it was!

# Compare train.py between any two points:

git diff HEAD~6 HEAD -- train.py

# Shows EVERY change to train.py across all six commits.


# --- SEARCHING HISTORY ---

# Find commits that mention "preprocessing":

git log --oneline --grep="preprocessing"

# >> f6f7g8h Add preprocessing pipeline with normalization
# >> xxxxxxx Integrate preprocessing into training pipeline

# Find commits that changed a specific FILE:

git log --oneline -- utils.py

# Shows only commits that touched utils.py.

# Find when a specific LINE was added (the "pickaxe"):

git log --oneline -S "normalize"

# Finds commits where the string "normalize" was added or removed.
# Super useful for "when did we introduce this function?"


# --- TIME TRAVEL ---

# Want to actually go back and look around?

git log --oneline

# Note the hash of your first commit. You can check it out:

git checkout HEAD~6

# Expected output:
# >> HEAD is now at a1a2b3c Add initial training script
# >> You are in 'detached HEAD' state...

# "Detached HEAD" = you're looking at an old commit, not a branch.
# It's safe — you're just looking, not changing anything.

cat train.py

# The original version!

ls

# Expected output:
# >> train.py

# Only train.py exists! utils.py, config.yaml, preprocess.py —
# none of them had been created yet at this point in history.

# Go back to the present:

git checkout main

ls

# Expected output:
# >> config.yaml  preprocess.py  train.py  utils.py

# Everything is back. Nothing was lost. Git is a time machine.


# --- VISUAL HISTORY ---

git log --oneline --graph --all --decorate

# Set up an alias so you don't type this every time:

git config --global alias.lg "log --oneline --graph --all --decorate"

# Now just type:

git lg

# You'll use this ALL the time.


# --- WHO CHANGED THIS LINE? (git blame) ---

git blame train.py

# Shows who last modified EACH LINE and in which commit.
# Incredibly useful for "who wrote this and why?"
# The commit hash on each line links you to the full context.



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 6: Undo Operations (git_reset_modes.png)         ║
# ║     Show the --soft / --mixed / --hard diagram. Then demo each.         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 5: "I Messed Up" — Undo Operations                     ~8 min     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Mistakes happen. Git has your back at every level.


# --- LEVEL 1: Undo changes to a file (not staged, not committed) ---

echo "THIS LINE BREAKS EVERYTHING" >> train.py
echo "ANOTHER BAD LINE" >> utils.py

git status

# Two files modified. Let's look at the damage:

git diff train.py

# You see the bad line. Discard changes to ONE specific file:

git restore train.py

git status

# >> modified: utils.py     (still modified)

# train.py is fixed, utils.py still has the bad line.
# Restore that one too:

git restore utils.py

git status

# >> nothing to commit, working tree clean

# Both files are back to their last committed versions.
#
# WARNING: git restore permanently throws away uncommitted changes.
#          There's no undo for this undo! Make sure you don't want them.


# --- LEVEL 2: Unstage a file (staged but not committed) ---

echo "temporary_debug = True" >> utils.py
echo "temp_flag = True" >> config.yaml

git add utils.py config.yaml
git status

# Expected output:
# >>   modified:   config.yaml   (GREEN — staged)
# >>   modified:   utils.py      (GREEN — staged)

# Oops, you meant to stage only utils.py, not config.yaml.
# Unstage config.yaml:

git restore --staged config.yaml
git status

# Expected output:
# >>   modified:   utils.py      (GREEN — staged)
# >>   modified:   config.yaml   (RED — modified but not staged)

# config.yaml is unstaged, but the change is still in the file.
# Clean everything up:

git restore --staged utils.py
git restore utils.py config.yaml


# --- LEVEL 3: Undo a commit ---

echo "# TODO: fix this later" >> train.py
git add train.py
git commit -m "Add bad TODO (we will undo this)"

git log --oneline -4

# See the bad commit at the top? Undo it:

git reset --soft HEAD~1

git status

# Expected output:
# >>   modified:   train.py    (GREEN — changes are still staged)

git log --oneline -4

# The commit is GONE from history, but the changes are still staged.
# You could now fix the code and re-commit, or just discard:

git restore --staged train.py
git restore train.py

# ╔══════════════════════════════════════════════════════════╗
# ║  THREE MODES OF git reset:                               ║
# ║                                                          ║
# ║  --soft    Undo commit, KEEP changes staged              ║
# ║            Use when: bad commit message, want to redo    ║
# ║                                                          ║
# ║  --mixed   Undo commit, KEEP changes unstaged (default)  ║
# ║            Use when: want to re-stage differently        ║
# ║                                                          ║
# ║  --hard    Undo commit, DELETE everything                ║
# ║            Use when: throw it all away (DANGEROUS!)      ║
# ╚══════════════════════════════════════════════════════════╝


# --- LEVEL 4: Undo a PUSHED commit (safely) ---
#
# When a commit has been pushed (others might have it), DON'T use reset.
# Use revert — it creates a NEW commit that undoes the old one.

echo "# Accidental debug code" >> train.py
git add train.py
git commit -m "Add accidental debug code"

git log --oneline -3

# Now "undo" this commit by creating a reversal:

git revert HEAD --no-edit

# Expected output:
# >> [main xxxxxxx] Revert "Add accidental debug code"

git log --oneline -4

# You'll see:
# >> xxxxxxx Revert "Add accidental debug code"    <-- the undo
# >> xxxxxxx Add accidental debug code              <-- the original

# The original commit is still in history (so others aren't confused),
# but its changes are undone. This is SAFE for shared branches.

cat train.py

# The debug code is gone.


# --- STASH: save work-in-progress ---

# Scenario: you're halfway through a change, but need to quickly
# check something on a clean working directory.

echo "# Work in progress: new loss function" >> train.py
echo "debug: true" >> config.yaml

git status

# Two modified files. You need a clean slate temporarily.

git stash

git status

# Expected output:
# >> nothing to commit, working tree clean

# Your changes vanished! But they're safe in the stash.

git stash list

# >> stash@{0}: WIP on main: xxxxxxx ...

# Do whatever you need on the clean directory...
# When ready, get your work back:

git stash pop

git status

# >> modified: config.yaml
# >> modified: train.py

# They're back! Stash is like a pocket for your WIP.

# Clean up:
git restore train.py config.yaml



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slides 7–8: Parallel Universes + Branch Pointer        ║
# ║     Show both diagrams. "A branch is just a pointer — instant, free."   ║
# ║     Then say "Let's try it."                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 6: Working on a Feature (Branching)                     ~8 min     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Advisor says: "Try adding data augmentation. But DON'T break what
# we have — I want to show the current version at a meeting tomorrow."
#
# Old you: copies the entire project folder.
# New you: creates a branch.
#
# A branch is a parallel universe. Experiment wildly in one universe
# while the other stays perfectly safe. A branch is literally just a
# pointer to a commit — creating one is instant and costs nothing.

# First, let's see what branch we're on:

git branch

# >> * main

# Create a new branch AND switch to it:

git checkout -b feature/augmentation

# Expected output:
# >> Switched to a new branch 'feature/augmentation'

git branch

# Expected output:
# >>   main
# >> * feature/augmentation

# The * shows your current branch. Let's verify:

git lg

# Notice: both main and feature/augmentation point to the SAME commit.
# The branch just created a new label — no files were copied.

# Now build the augmentation feature:

cat > augment.py << 'EOF'
import numpy as np

def add_noise(X, scale=0.1):
    """Add Gaussian noise for data augmentation."""
    noise = np.random.normal(0, scale, X.shape)
    return X + noise

def random_flip(X, p=0.5):
    """Randomly flip features with probability p."""
    mask = np.random.random(X.shape) < p
    return X * (1 - 2 * mask)

def oversample(X, y, minority_class=1, factor=2):
    """Oversample the minority class."""
    minority_mask = y == minority_class
    X_minority = X[minority_mask]
    y_minority = y[minority_mask]
    X_aug = np.tile(X_minority, (factor, 1))
    y_aug = np.tile(y_minority, factor)
    return np.vstack([X, X_aug]), np.concatenate([y, y_aug])
EOF

git add augment.py
git commit -m "Add data augmentation with noise, flip, and oversample"

# Update train.py to use augmentation:

cat > train.py << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression
from preprocess import load_and_preprocess
from augment import add_noise
from utils import accuracy

def train(X, y):
    X_aug = add_noise(X)
    X_combined = np.vstack([X, X_aug])
    y_combined = np.concatenate([y, y])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_combined, y_combined)
    return model

def evaluate(model, X, y):
    preds = model.predict(X)
    acc = accuracy(y, preds)
    print(f"Accuracy: {acc:.3f}")
    return acc

if __name__ == "__main__":
    X, y = load_and_preprocess("data/train.npy")
    model = train(X, y)
    evaluate(model, X, y)
EOF

git add train.py
git commit -m "Integrate augmentation into training pipeline"

git lg

# You'll see feature/augmentation is 2 commits ahead of main.
# main hasn't moved — it still points to where it was.


# --- THE MAGIC MOMENT ---

git checkout main

ls *.py

# Expected output:
# >> preprocess.py  train.py  utils.py

# WHERE DID augment.py GO?!
# It's NOT deleted. It's safely on the feature/augmentation branch.
# Git swapped your working directory to match main's snapshot.

cat train.py

# It's the OLD version! No augmentation imports, no add_noise().
# Your advisor can demo this version at their meeting.
# Two parallel universes, one folder.

# Prove the feature branch is safe:

git checkout feature/augmentation
ls *.py

# >> augment.py  preprocess.py  train.py  utils.py

# It's all there! Let's also check that the branches are truly separate:

git lg

# You can see both pointers clearly — main behind, feature ahead.

# Switch back to main for merging:

git checkout main



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 9: Diverging Branches                            ║
# ║     Show the diagram. "When both branches have new commits..."          ║
# ║     Then demo fast-forward and three-way merge.                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 7: Merging (Bringing Branches Together)                ~10 min     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Advisor says: "The augmentation feature is good. Merge it in."


# --- FAST-FORWARD MERGE ---
#
# When main has NO new commits since you branched, merging is trivial.
# Git just moves the main pointer forward.

git merge feature/augmentation

# Expected output:
# >> Updating xxxxxxx..xxxxxxx
# >> Fast-forward
# >>  augment.py | 24 ++++++++++++++++++++++++
# >>  train.py   |  ... changed ...

# "Fast-forward" = main just moved its pointer. No merge commit needed.

ls *.py

# >> augment.py  preprocess.py  train.py  utils.py

git lg

# main and feature/augmentation now point to the same commit.

# Clean up the merged branch (optional but good practice):

git branch -d feature/augmentation

# >> Deleted branch feature/augmentation

git branch

# >> * main

# The commits are still in history — the branch LABEL is just removed.


# --- THREE-WAY MERGE ---
#
# What if main ALSO got changes while the feature was being developed?
# Then Git can't just fast-forward — it needs to create a merge commit.

# Create a branch for evaluation:

git checkout -b feature/evaluation

cat > evaluate.py << 'EOF'
import numpy as np
from utils import accuracy, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Full evaluation with metrics."""
    preds = model.predict(X_test)
    acc = accuracy(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"Test Accuracy:     {acc:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    return {"accuracy": acc, "confusion_matrix": cm}
EOF

git add evaluate.py
git commit -m "Add evaluation module with full metrics"

# Meanwhile, go back to main and make a DIFFERENT change:

git checkout main

cat > README.md << 'EOF'
# ML Project

A machine learning pipeline with data augmentation and preprocessing.

## Files
- `train.py` — Training with augmentation
- `preprocess.py` — Data loading and normalization
- `augment.py` — Data augmentation (noise, flip, oversample)
- `evaluate.py` — Model evaluation
- `utils.py` — Utility functions
- `config.yaml` — Configuration

## Usage
```bash
python preprocess.py
python train.py
```
EOF

git add README.md
git commit -m "Add README with project documentation"

# Now the branches have DIVERGED:

git lg

# Expected output (roughly):
# >> * xxxxxxx (HEAD -> main) Add README with project documentation
# >> | * xxxxxxx (feature/evaluation) Add evaluation module with full metrics
# >> |/
# >> * xxxxxxx Integrate augmentation into training pipeline

# main has README.md, feature/evaluation has evaluate.py.
# Both changed since they split. Git needs a THREE-WAY merge:

git merge feature/evaluation -m "Merge evaluation module into main"

# Expected output:
# >> Merge made by the 'ort' strategy.
# >>  evaluate.py | 14 ++++++++++++++
# >>  1 file changed, 14 insertions(+)

ls

# >> README.md  augment.py  config.yaml  evaluate.py  preprocess.py  train.py  utils.py

# Both README.md AND evaluate.py are here! Git wove the branches together.

git lg

# You'll see the merge commit with two parent lines converging.
# The merge commit says "I combined work from both branches."

# Look at the merge commit in detail:

git show HEAD --stat

# You can see it has TWO parents — that's what makes it a merge commit.

# Clean up:
git branch -d feature/evaluation



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Stay on Slide 9 (or hide — terminal only)              ║
# ║     Conflicts are best understood live. Stay in terminal.               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 8: Merge Conflicts                                     ~10 min     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# So far merges went smoothly because changes were in DIFFERENT files.
# But what happens when two branches change the SAME lines?
#
# Scenario: You and a colleague both want to tune hyperparameters.
# You each create a branch and edit config.yaml differently.


# --- SETUP: Create two conflicting branches ---

# Branch 1: You want a higher learning rate and more iterations.

git checkout -b experiment/high-lr

cat > config.yaml << 'EOF'
model:
  type: logistic_regression
  max_iter: 2000
  learning_rate: 0.01
  normalize: true

data:
  train_path: data/train.npy
  test_path: data/test.npy
  validation_split: 0.2
EOF

git add config.yaml
git commit -m "Try high learning rate with more iterations"

# Branch 2: Your colleague wants fewer iterations and lower LR.

git checkout main
git checkout -b experiment/low-lr

cat > config.yaml << 'EOF'
model:
  type: logistic_regression
  max_iter: 500
  learning_rate: 0.001
  normalize: true

data:
  train_path: data/train.npy
  test_path: data/test.npy
  validation_split: 0.2
EOF

git add config.yaml
git commit -m "Try low learning rate with fewer iterations"

git lg

# Both branches changed config.yaml differently. Now what?


# --- TRIGGER THE CONFLICT ---

git checkout main

# First merge goes fine (main hadn't changed config.yaml):

git merge experiment/high-lr -m "Merge high-lr experiment"

# Now try the second:

git merge experiment/low-lr

# Expected output:
# >> Auto-merging config.yaml
# >> CONFLICT (content): Merge conflict in config.yaml
# >> Automatic merge failed; fix conflicts and then commit the result.

# DON'T PANIC! This is normal. Git is saying:
# "Two people changed the same lines. I need a human to decide."


# --- INSPECT THE CONFLICT ---

git status

# Expected output:
# >> Unmerged paths:
# >>   (use "git add <file>..." to mark resolution)
# >>         both modified:   config.yaml

cat config.yaml

# You'll see conflict markers:
# >> model:
# >>   type: logistic_regression
# >> <<<<<<< HEAD
# >>   max_iter: 2000
# >>   learning_rate: 0.01
# >> =======
# >>   max_iter: 500
# >>   learning_rate: 0.001
# >> >>>>>>> experiment/low-lr
# >>   normalize: true
# >> ...

# How to read this:
#   <<<<<<< HEAD        = YOUR branch's version (main, which has high-lr)
#   =======             = separator
#   >>>>>>> branch      = the OTHER branch's version
#
# Lines OUTSIDE the markers were merged successfully (no conflict there).
# Only the lines BETWEEN markers are disputed.


# --- RESOLVE THE CONFLICT ---

# Open config.yaml in your editor. You have three choices:
#   1. Keep YOUR version (delete theirs + markers)
#   2. Keep THEIR version (delete yours + markers)
#   3. Combine / compromise (write something new)
#
# Let's compromise: max_iter=1000 and learning_rate=0.005

cat > config.yaml << 'EOF'
model:
  type: logistic_regression
  max_iter: 1000
  learning_rate: 0.005
  normalize: true

data:
  train_path: data/train.npy
  test_path: data/test.npy
  validation_split: 0.2
EOF

# IMPORTANT: make sure ALL markers (<<<, ===, >>>) are removed!
# Now tell Git the conflict is resolved:

git add config.yaml

git status

# >> All conflicts fixed but you are still merging.

git commit -m "Resolve conflict: compromise on hyperparameters (max_iter=1000, lr=0.005)"

git lg

# The merge commit records your decision. Both experiment branches
# are integrated, and the config has your chosen values.
#
# TIP: If you get overwhelmed during a conflict:
#   git merge --abort
# Cancels the merge entirely. No harm done.

# Clean up experiment branches:
git branch -d experiment/high-lr
git branch -d experiment/low-lr


# --- BONUS: Conflict in a Python file ---
#
# Conflicts aren't just in config files. Let's see one in Python.

git checkout -b feature/train-v2

cat > train.py << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from preprocess import load_and_preprocess
from augment import add_noise
from utils import accuracy

def train(X, y, cv_folds=5):
    X_aug = add_noise(X)
    X_combined = np.vstack([X, X_aug])
    y_combined = np.concatenate([y, y])
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X_combined, y_combined, cv=cv_folds)
    print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    model.fit(X_combined, y_combined)
    return model

def evaluate(model, X, y):
    preds = model.predict(X)
    acc = accuracy(y, preds)
    print(f"Accuracy: {acc:.3f}")
    return acc

if __name__ == "__main__":
    X, y = load_and_preprocess("data/train.npy")
    model = train(X, y)
    evaluate(model, X, y)
EOF

git add train.py
git commit -m "Add cross-validation to training"

git checkout main
git checkout -b feature/train-v3

cat > train.py << 'EOF'
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess
from augment import add_noise
from utils import accuracy

def train(X, y, model_type="logistic"):
    X_aug = add_noise(X)
    X_combined = np.vstack([X, X_aug])
    y_combined = np.concatenate([y, y])
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100)
    model.fit(X_combined, y_combined)
    return model

def evaluate(model, X, y):
    preds = model.predict(X)
    acc = accuracy(y, preds)
    print(f"Accuracy: {acc:.3f}")
    return acc

if __name__ == "__main__":
    X, y = load_and_preprocess("data/train.npy")
    model = train(X, y, model_type="rf")
    evaluate(model, X, y)
EOF

git add train.py
git commit -m "Add random forest option to training"

git checkout main
git merge feature/train-v2 -m "Merge cross-validation feature"

git merge feature/train-v3

# CONFLICT in train.py! Both branches changed the train() function.

git status

# >> both modified: train.py

cat train.py

# You'll see markers around the train() function and imports.
# In a real project, you'd carefully combine both changes:
# cross-validation AND model selection. For now, pick one:

git checkout --theirs train.py

# This takes train-v3's version entirely. You could also use:
#   git checkout --ours train.py     (take our version)

git add train.py
git commit -m "Resolve conflict: keep random forest option"

git branch -d feature/train-v2
git branch -d feature/train-v3



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Hide slides — terminal only for this section            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 9: Ignoring Files That Don't Belong (.gitignore)        ~5 min     │
# └──────────────────────────────────────────────────────────────────────────┘
#
# As your project grows, junk files appear everywhere.

# Simulate typical ML project clutter:

mkdir -p __pycache__ data logs .vscode
echo "cached bytecode" > __pycache__/train.cpython-310.pyc
echo "cached bytecode" > __pycache__/utils.cpython-310.pyc
echo "SECRET_API_KEY=sk-abc123" > .env
echo "WANDB_KEY=wk-secret456" >> .env
dd if=/dev/zero bs=1024 count=100 of=model.pkl 2>/dev/null
dd if=/dev/zero bs=1024 count=50 of=model_v2.h5 2>/dev/null
echo "1,2,3" > data/train.csv
echo "4,5,6" > data/test.csv
echo "epoch,loss" > logs/training.log
echo '{"editor.fontSize": 14}' > .vscode/settings.json
touch .DS_Store

git status

# Expected output includes:
# >>   .DS_Store               <-- OS junk
# >>   .env                    <-- SECRETS! API keys!
# >>   .vscode/                <-- Personal editor settings
# >>   __pycache__/            <-- Python cache (regenerated)
# >>   data/                   <-- Could be huge datasets
# >>   logs/                   <-- Experiment logs
# >>   model.pkl               <-- Trained model (100KB+)
# >>   model_v2.h5             <-- Another model file

# Git wants to track ALL of this. Let's tell it what to ignore:

cat > .gitignore << 'EOF'
# Python bytecode (regenerated from source code)
__pycache__/
*.pyc

# Secrets — NEVER commit API keys, passwords, tokens!
.env
*.secret

# Large model files (use Git LFS or DVC instead)
*.pkl
*.h5
*.pt
*.onnx

# Data files (too large for Git)
data/*.csv
data/*.parquet
data/*.npy

# Experiment logs (tracked by wandb/mlflow)
logs/
wandb/
mlruns/

# IDE and editor settings (personal preference)
.vscode/
.idea/
*.swp
*~

# OS-generated files
.DS_Store
Thumbs.db
EOF

# Check the difference:

git status

# Expected output:
# >>   .gitignore

# ONLY .gitignore shows up! Everything else is hidden.

git add .gitignore
git commit -m "Add .gitignore for ML project"


# --- WHAT IF YOU ALREADY COMMITTED A SECRET? ---

# Let's say you accidentally committed .env BEFORE adding .gitignore:
# (We didn't, but it's a common mistake)
#
# Changing or deleting .env does NOT remove it from Git history!
# Anyone who clones the repo can find it with:
#   git log --all --full-history -- .env
#
# If this happens:
#   1. ROTATE THE SECRET IMMEDIATELY (generate new API key)
#   2. Add it to .gitignore
#   3. Remove from tracking: git rm --cached .env
#   4. Consider scrubbing history with BFG Repo Cleaner


# --- WHAT IF .gitignore ISN'T WORKING? ---

# .gitignore only ignores UNTRACKED files. If a file is already
# tracked (you committed it before), .gitignore won't help.
# You need to un-track it:
#
#   git rm --cached <file>
#   git commit -m "Stop tracking <file>"
#
# Then .gitignore will apply.



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 10: Local ↔ Remote (git_local_remote.png)        ║
# ║     Show the diagram. "Everything is local until push/pull."            ║
# ║     Then demo push, clone, pull.                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ACT 10: Sharing Your Work (Remotes)                         ~10 min    │
# └──────────────────────────────────────────────────────────────────────────┘
#
# So far everything is local. What if your laptop dies?
# What if your teammate needs your code? You need a REMOTE.
#
# In real life, the remote is GitHub/GitLab. For this demo,
# we'll simulate one with a local "bare" repository.


# --- 10.1 Create a "GitHub" ---

cd ..
git init --bare fake-github.git

# Expected output:
# >> Initialized empty Git repository in /path/to/fake-github.git/

# A "bare" repo has no working directory — just the Git database.
# This is exactly what GitHub/GitLab stores on their servers.


# --- 10.2 Connect your project to the remote ---

cd ml-project
git remote add origin ../fake-github.git

# "origin" is the conventional name for your primary remote.
# Let's verify:

git remote -v

# Expected output:
# >> origin  ../fake-github.git (fetch)
# >> origin  ../fake-github.git (push)


# --- 10.3 Push your work ---

git push -u origin main

# Expected output:
# >> To ../fake-github.git
# >>  * [new branch]      main -> main
# >> Branch 'main' set up to track remote branch 'main' from 'origin'.

# All your commits are now on the remote!
# The -u flag sets up tracking — future push/pull just need "git push".


# --- 10.4 A collaborator joins the project ---

cd ..
git clone fake-github.git collaborator-clone

# Expected output:
# >> Cloning into 'collaborator-clone'...

cd collaborator-clone
git config user.name "Alice (Collaborator)"
git config user.email "alice@iitgn.ac.in"

# Alice has the ENTIRE history:

git lg

# She sees everything — all commits, all history. Let's verify:

git log --oneline | wc -l

# Same number of commits as your repo!


# --- 10.5 Alice makes a change and pushes ---

echo "" >> README.md
echo "## Getting Started" >> README.md
echo '```bash' >> README.md
echo "pip install numpy scikit-learn" >> README.md
echo '```' >> README.md
echo "" >> README.md
echo "## Contributing" >> README.md
echo "1. Fork the repo" >> README.md
echo "2. Create a feature branch" >> README.md
echo "3. Make your changes and commit" >> README.md
echo "4. Push and open a pull request" >> README.md

git add README.md
git commit -m "Add getting-started and contributing sections to README"
git push origin main

# Alice's commit is now on the remote.


# --- 10.6 You pull Alice's change ---

cd ../ml-project

# First, let's see what we have:

cat README.md

# No "Getting Started" section. Alice's change isn't here yet.

git pull origin main

# Expected output:
# >> From ../fake-github.git
# >>    xxxxxxx..xxxxxxx  main     -> origin/main
# >> Updating xxxxxxx..xxxxxxx
# >> Fast-forward
# >>  README.md | 10 ++++++++++

cat README.md

# Alice's sections are here! That's collaboration.


# --- 10.7 What if you BOTH push? ---

# Let's simulate a collision.

# Alice adds a requirements file:
cd ../collaborator-clone
cat > requirements.txt << 'EOF'
numpy>=1.21
scikit-learn>=1.0
EOF

git add requirements.txt
git commit -m "Add requirements.txt"
git push origin main

# Meanwhile, YOU also made a change (without pulling first):
cd ../ml-project
echo "# Experiment tracking: wandb" >> README.md
git add README.md
git commit -m "Add experiment tracking note to README"

# Now try to push:

git push origin main

# Expected output:
# >> ! [rejected]        main -> main (fetch first)
# >> error: failed to push some refs to '../fake-github.git'
# >> hint: Updates were rejected because the remote contains work that you do
# >> hint: not have locally.

# Git REJECTED your push! It's protecting you.
# You need to pull first (get Alice's changes), then push.

git pull origin main

# If there's no conflict, Git auto-merges. If there IS a conflict,
# you'd resolve it like in Act 8.

# Expected output:
# >> Merge made by the 'ort' strategy.
# >>  requirements.txt | 2 ++

# Now push your combined work:

git push origin main

git lg

# You'll see the merge commit that combined your work with Alice's.

# ╔══════════════════════════════════════════════════════════╗
# ║  REMOTE COMMANDS:                                        ║
# ║                                                          ║
# ║  git clone <url>    Copy entire repo (first time)        ║
# ║  git remote -v      Show connected remotes               ║
# ║  git push           Upload your commits to remote        ║
# ║  git pull           Download + merge remote commits      ║
# ║  git fetch          Download only (don't merge yet)      ║
# ║                                                          ║
# ║  GOLDEN RULE: Always pull before you push!               ║
# ╚══════════════════════════════════════════════════════════╝



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Hide slides — terminal only                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  EPILOGUE: Good Commit Messages                              ~3 min     │
# └──────────────────────────────────────────────────────────────────────────┘

# Before we wrap up — commit messages matter more than you think.
# Six months from now, you'll search your history to understand
# why something changed. Good messages save hours.

# BAD messages — tell you nothing:
# >> git commit -m "fix"
# >> git commit -m "stuff"
# >> git commit -m "update"
# >> git commit -m "asdfgh"
# >> git commit -m "WIP"

# GOOD messages — complete the sentence "If applied, this commit will...":
# >> git commit -m "Fix off-by-one error in batch data loader"
# >> git commit -m "Add data augmentation with Gaussian noise"
# >> git commit -m "Remove unused imports from utils.py"
# >> git commit -m "Increase max_iter to fix convergence warning"

# For bigger changes, use a multi-line message:
# >> git commit -m "Add cross-validation to training pipeline
# >>
# >> - Use 5-fold stratified CV by default
# >> - Print mean and std of accuracy across folds
# >> - Keep final model trained on full dataset"

# Look at our history with these good messages:

git log --oneline

# Each line tells you exactly what happened. You can navigate
# months of work in seconds.



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  📽  PROJECTOR → Slide 11: Quick Reference                               ║
# ║     Show the cheat sheet. Leave it up while students ask questions.      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  THE END                                                                 │
# └──────────────────────────────────────────────────────────────────────────┘
#
# You started with train_v2_FINAL_fixed_actually_final.py.
# You ended with a clean repo, branches, merges, and remote collaboration.

cd ../ml-project
git lg

# Your directories:
#   ml-project/              — your repo
#   fake-github.git/         — the "remote"
#   collaborator-clone/      — Alice's clone



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QUICK REFERENCE                                                        ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║                                                                         ║
# ║  DAILY WORKFLOW                                                         ║
# ║    git status                what changed?                              ║
# ║    git diff                  what exactly changed?                      ║
# ║    git diff --staged         what am I about to commit?                 ║
# ║    git add <files>           stage changes                              ║
# ║    git commit -m "why"       save a checkpoint                          ║
# ║    git push                  upload to remote                           ║
# ║    git pull                  download from remote                       ║
# ║                                                                         ║
# ║  BRANCHING                                                              ║
# ║    git branch                list all branches                          ║
# ║    git checkout -b <name>    create + switch to branch                  ║
# ║    git checkout <name>       switch to existing branch                  ║
# ║    git merge <branch>        merge branch into current                  ║
# ║    git branch -d <name>      delete merged branch                      ║
# ║                                                                         ║
# ║  HISTORY                                                                ║
# ║    git log --oneline         compact history                            ║
# ║    git lg                    visual graph (after alias setup)           ║
# ║    git show <hash>           inspect any commit                         ║
# ║    git diff <A> <B>          compare any two points                     ║
# ║    git blame <file>          who changed each line?                     ║
# ║    git log --grep="text"     search commit messages                     ║
# ║    git log -S "code"         search for code changes                    ║
# ║    git log -- <file>         history of a specific file                 ║
# ║                                                                         ║
# ║  UNDO                                                                   ║
# ║    git restore <file>            discard file changes                   ║
# ║    git restore --staged <file>   unstage a file                         ║
# ║    git reset --soft HEAD~1       undo commit (keep staged)              ║
# ║    git reset HEAD~1              undo commit (keep unstaged)            ║
# ║    git reset --hard HEAD~1       undo commit (DELETE all) [DANGER]      ║
# ║    git revert <hash>             undo pushed commit (safe)              ║
# ║    git stash / stash pop         save/restore work-in-progress          ║
# ║    git merge --abort             cancel a merge in progress             ║
# ║    git checkout --ours <file>    take our version in conflict           ║
# ║    git checkout --theirs <file>  take their version in conflict         ║
# ║                                                                         ║
# ║  REMOTES                                                                ║
# ║    git clone <url>               copy entire repo                       ║
# ║    git remote -v                 show connected remotes                 ║
# ║    git push -u origin main       push + set up tracking                 ║
# ║    git pull origin main          download + merge                       ║
# ║    git fetch origin              download only (no merge)               ║
# ║                                                                         ║
# ║  COMMIT MESSAGE RULE                                                    ║
# ║    Complete: "If applied, this commit will ___."                        ║
# ║    Good: "Add data augmentation for training images"                    ║
# ║    Good: "Fix off-by-one error in batch loader"                         ║
# ║    Bad:  "fix", "stuff", "update", "asdfgh"                            ║
# ║                                                                         ║
# ║  BEST PRACTICES                                                         ║
# ║    1. Commit early, commit often                                        ║
# ║    2. Write meaningful commit messages                                  ║
# ║    3. Never commit secrets (.env, API keys)                             ║
# ║    4. Never commit large files (models, data) — use DVC/LFS            ║
# ║    5. Branch for every experiment                                       ║
# ║    6. Always git status before git add                                  ║
# ║    7. Pull before you push                                              ║
# ║    8. Delete branches after merging                                     ║
# ║                                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

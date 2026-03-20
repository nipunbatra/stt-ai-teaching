#!/bin/bash
# Week 09: Git Deep-Dive — Follow-Along Demo
# ============================================
# The instructor runs this on the projector. Students type the same
# commands on their own laptops. The script shows each command, pauses
# so everyone can type it, then runs it to show expected output.
#
# Modes:
#   ./git_demo.sh              Follow-along (default) — pauses after each command
#   ./git_demo.sh --auto       Auto-run — no pauses, for replay/review
#
# The script creates a temporary directory. Nothing is cleaned up
# so students can explore the result afterward.

set -e

# ─── Mode ─────────────────────────────────────────────────────────────────────

AUTO=false
if [ "$1" = "--auto" ]; then
    AUTO=true
fi

# ─── Colors & Helpers ─────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

slide_ref() {
    printf "\n"
    printf "${RED}╔══════════════════════════════════════════════════════════╗${NC}\n"
    printf "${RED}║  SLIDE CUE: %s${NC}\n" "$1"
    printf "${RED}╚══════════════════════════════════════════════════════════╝${NC}\n"
    printf "\n"
}

section() {
    printf "\n"
    printf "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "${BOLD}${BLUE}  %s${NC}\n" "$1"
    printf "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "\n"
}

step() {
    printf "${CYAN}▸ %s${NC}\n" "$1"
}

explain() {
    printf "${DIM}  # %s${NC}\n" "$1"
}

# Show a command, pause so students can type it, then run it.
run_cmd() {
    printf "\n"
    printf "${GREEN}  TYPE ▶  ${NC}${YELLOW}%s${NC}\n" "$1"
    if [ "$AUTO" = false ]; then
        printf "${DIM}  (press Enter after you've typed it)${NC}"
        read -r _unused
    fi
    printf "${DIM}  ── output ──${NC}\n"
    eval "$1"
    printf "\n"
}

# Run a command silently (setup, not for students to type).
run_silent() {
    eval "$1"
}

pause() {
    if [ "$AUTO" = false ]; then
        printf "\n"
        printf "${GREEN}── Ready for next section? Press Enter ──${NC}"
        read -r _unused
        printf "\n"
    fi
}

show_state() {
    printf "${DIM}── Current state ──${NC}\n"
    printf "${GREEN}  TYPE ▶  ${NC}${YELLOW}git status${NC}\n"
    if [ "$AUTO" = false ]; then
        printf "${DIM}  (press Enter after you've typed it)${NC}"
        read -r _unused
    fi
    git status
    printf "\n"
    printf "${GREEN}  TYPE ▶  ${NC}${YELLOW}git log --oneline --graph --all --decorate${NC}\n"
    if [ "$AUTO" = false ]; then
        printf "${DIM}  (press Enter after you've typed it)${NC}"
        read -r _unused
    fi
    git log --oneline --graph --all --decorate 2>/dev/null || printf "  (no commits yet)\n"
    printf "\n"
}

# ─── Setup: create a temp directory ───────────────────────────────────────────

DEMO_DIR=$(mktemp -d "${TMPDIR:-/tmp}/git-demo-XXXXXX")
printf "${BOLD}Git Deep-Dive Demo${NC}\n"
printf "Working in: ${CYAN}%s${NC}\n\n" "$DEMO_DIR"
cd "$DEMO_DIR"

# =============================================================================
# SECTION 1: Init & First Commits
# =============================================================================

slide_ref "Advance to slide 7 — 'Demo §1: Let's Build an ML Project'"
section "1. INIT & FIRST COMMITS"

step "Create a new Git repository"
run_cmd "git init ml-project"
cd ml-project

explain "Configure user and default branch for this demo repo"
run_silent 'git config user.name "Demo Student"'
run_silent 'git config user.email "student@iitgn.ac.in"'
run_silent 'git checkout -b main 2>/dev/null || true'

pause

step "Create our first file: train.py"
explain "Open your editor and create train.py with this content:"
printf "${DIM}────────────────────────────────────────${NC}\n"
cat << 'SHOW'
import numpy as np
from sklearn.linear_model import LogisticRegression

def load_data(path):
    """Load and split dataset."""
    data = np.load(path)
    X, y = data[:, :-1], data[:, -1]
    return X, y

def train(X, y):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    X, y = load_data("data/train.npy")
    model = train(X, y)
    print(f"Accuracy: {model.score(X, y):.3f}")
SHOW
printf "${DIM}────────────────────────────────────────${NC}\n"

# Actually create the file
cat > train.py << 'PYEOF'
import numpy as np
from sklearn.linear_model import LogisticRegression

def load_data(path):
    """Load and split dataset."""
    data = np.load(path)
    X, y = data[:, :-1], data[:, -1]
    return X, y

def train(X, y):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    X, y = load_data("data/train.npy")
    model = train(X, y)
    print(f"Accuracy: {model.score(X, y):.3f}")
PYEOF

pause

step "Check git status — Git sees the new file"
run_cmd "git status"

step "Stage the file"
run_cmd "git add train.py"

step "Check status again — file is now staged (green)"
run_cmd "git status"

pause

step "Make our first commit!"
run_cmd 'git commit -m "Add training script with logistic regression"'

show_state

pause

step "Now create two more files: utils.py and config.yaml"
explain "Create utils.py:"
printf "${DIM}────────────────────────────────────────${NC}\n"
cat << 'SHOW'
import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])
SHOW
printf "${DIM}────────────────────────────────────────${NC}\n"
explain "Create config.yaml:"
printf "${DIM}────────────────────────────────────────${NC}\n"
cat << 'SHOW'
model:
  type: logistic_regression
  max_iter: 1000

data:
  train_path: data/train.npy
  test_path: data/test.npy
SHOW
printf "${DIM}────────────────────────────────────────${NC}\n"

# Actually create the files
cat > utils.py << 'PYEOF'
import numpy as np

def accuracy(y_true, y_pred):
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    """Simple 2x2 confusion matrix."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])
PYEOF

cat > config.yaml << 'EOF'
model:
  type: logistic_regression
  max_iter: 1000

data:
  train_path: data/train.npy
  test_path: data/test.npy
EOF

pause

step "Stage and commit both files at once"
run_cmd "git add utils.py config.yaml"
run_cmd 'git commit -m "Add utils module and config file"'

show_state

pause

step "Modify train.py — add import from utils and update main block"
explain "Edit train.py: add 'from utils import accuracy, confusion_matrix' and use them"

# Actually update the file
cat > train.py << 'PYEOF'
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import accuracy, confusion_matrix

def load_data(path):
    """Load and split dataset."""
    data = np.load(path)
    X, y = data[:, :-1], data[:, -1]
    return X, y

def train(X, y):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    X, y = load_data("data/train.npy")
    model = train(X, y)
    preds = model.predict(X)
    print(f"Accuracy: {accuracy(y, preds):.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, preds)}")
PYEOF

step "See exactly what changed"
run_cmd "git diff train.py"

pause

step "Stage and commit the modification"
run_cmd "git add train.py"
run_cmd 'git commit -m "Use utils functions in training script"'

show_state

pause

# =============================================================================
# SECTION 2: Branching (Fast-Forward Merge)
# =============================================================================

slide_ref "Advance to slide 11 — 'Demo §2: Branching & Fast-Forward Merge'"
section "2. BRANCHING & FAST-FORWARD MERGE"

step "Create and switch to a feature branch"
run_cmd "git checkout -b feature/augmentation"

show_state

pause

step "Create augment.py on the feature branch"
explain "Create augment.py with add_noise() and oversample() functions"

cat > augment.py << 'PYEOF'
import numpy as np

def add_noise(X, scale=0.1):
    """Add Gaussian noise for data augmentation."""
    noise = np.random.normal(0, scale, X.shape)
    return X + noise

def oversample(X, y, minority_class=1, factor=2):
    """Oversample the minority class."""
    minority_mask = y == minority_class
    X_minority = X[minority_mask]
    y_minority = y[minority_mask]
    X_aug = np.tile(X_minority, (factor, 1))
    y_aug = np.tile(y_minority, factor)
    return np.vstack([X, X_aug]), np.concatenate([y, y_aug])
PYEOF

run_cmd "git add augment.py"
run_cmd 'git commit -m "Add data augmentation utilities"'

show_state
explain "feature/augmentation is 1 commit ahead of main"

pause

step "Switch back to main — watch augment.py DISAPPEAR!"
run_cmd "git checkout main"
run_cmd "ls *.py"
explain "augment.py only exists on the feature branch!"

pause

step "Merge feature branch into main (fast-forward)"
run_cmd "git merge feature/augmentation"
explain "Fast-forward: main just moved its pointer forward. No merge commit needed."

show_state

step "Now augment.py is on main too"
run_cmd "ls *.py"

pause

# =============================================================================
# SECTION 3: Three-Way Merge
# =============================================================================

slide_ref "Advance to slide 13 — 'Demo §3: Three-Way Merge'"
section "3. THREE-WAY MERGE (Diverging Branches)"

step "Create a new feature branch"
run_cmd "git checkout -b feature/evaluation"

step "Create evaluate.py on the feature branch"
cat > evaluate.py << 'PYEOF'
import numpy as np
from utils import accuracy, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data."""
    preds = model.predict(X_test)
    acc = accuracy(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    return acc
PYEOF

run_cmd "git add evaluate.py"
run_cmd 'git commit -m "Add evaluation module"'

pause

step "Switch to main and make a DIFFERENT change"
run_cmd "git checkout main"

explain "Create README.md on main"
cat > README.md << 'EOF'
# ML Project

A machine learning pipeline for binary classification.

## Files
- `train.py` - Training script
- `utils.py` - Utility functions
- `augment.py` - Data augmentation
- `config.yaml` - Configuration
EOF

run_cmd "git add README.md"
run_cmd 'git commit -m "Add README documentation"'

step "Now the branches have diverged!"
show_state
explain "main has README.md, feature/evaluation has evaluate.py"

pause

step "Three-way merge: bring evaluation into main"
run_cmd "git merge feature/evaluation -m 'Merge evaluation module into main'"

show_state
explain "A merge commit was created with TWO parents!"

run_cmd "ls"
explain "Both README.md AND evaluate.py are here now"

pause

# =============================================================================
# SECTION 4: Merge Conflict
# =============================================================================

slide_ref "Advance to slide 15 — 'Demo §4: Create & Resolve a Conflict'"
section "4. MERGE CONFLICT"

step "Create two branches that will conflict on config.yaml"
run_cmd "git checkout -b experiment/high-lr"

explain "Change max_iter to 2000 on this branch"
sed -i.bak 's/max_iter: 1000/max_iter: 2000/' config.yaml && rm -f config.yaml.bak
printf "  learning_rate: 0.01\n" >> config.yaml
run_cmd "git add config.yaml"
run_cmd 'git commit -m "Increase max_iter and set high learning rate"'

run_cmd "git checkout main"
run_cmd "git checkout -b experiment/low-lr"

explain "Change max_iter to 500 on THIS branch (will conflict!)"
sed -i.bak 's/max_iter: 1000/max_iter: 500/' config.yaml && rm -f config.yaml.bak
printf "  learning_rate: 0.001\n" >> config.yaml
run_cmd "git add config.yaml"
run_cmd 'git commit -m "Decrease max_iter and set low learning rate"'

show_state

pause

step "Switch to main and merge high-lr first (no conflict)"
run_cmd "git checkout main"
run_cmd "git merge experiment/high-lr -m 'Merge high learning rate experiment'"

step "Now try to merge low-lr — CONFLICT!"
printf "\n${RED}  This will cause a conflict:${NC}\n"
git merge experiment/low-lr || true
printf "\n"

step "Git shows us the conflict"
run_cmd "git status"

step "Look at the conflicted file — see the conflict markers"
run_cmd "cat config.yaml"

pause

step "Resolve the conflict manually"
explain "Edit config.yaml: remove markers, pick max_iter: 1000 and lr: 0.005"

cat > config.yaml << 'EOF'
model:
  type: logistic_regression
  max_iter: 1000

data:
  train_path: data/train.npy
  test_path: data/test.npy
  learning_rate: 0.005
EOF

run_cmd "git add config.yaml"
run_cmd 'git commit -m "Resolve conflict: compromise on hyperparameters"'

show_state
explain "Conflict resolved! The merge commit records our decision."

pause

# =============================================================================
# SECTION 5: History Exploration
# =============================================================================

slide_ref "Advance to slide 17 — 'Demo §5: Exploring History'"
section "5. HISTORY EXPLORATION"

step "View the full commit graph"
run_cmd "git log --oneline --graph --all --decorate"

pause

step "View details of a specific commit"
LATEST=$(git log --oneline -1 | cut -d' ' -f1)
run_cmd "git show $LATEST"

pause

step "Compare two commits"
FIRST=$(git log --oneline --reverse | head -1 | cut -d' ' -f1)
run_cmd "git diff $FIRST HEAD -- train.py"
explain "Shows all changes to train.py from the first commit to now"

pause

step "Search commit messages"
run_cmd 'git log --oneline --grep="augment"'

pause

# =============================================================================
# SECTION 6: Undo Operations
# =============================================================================

slide_ref "Advance to slide 19 — 'Demo §6: Undo in Action'"
section "6. UNDO OPERATIONS"

step "Modify a file, then UNDO the change"
explain "Add a bad line to train.py"
printf "# OOPS I broke something\n" >> train.py
run_cmd "git diff train.py"

step "Restore the file to its last committed version"
run_cmd "git restore train.py"
run_cmd "git diff train.py"
explain "Empty diff — the bad change is gone!"

pause

step "Stage a file, then UNSTAGE it"
printf "temporary = True\n" >> utils.py
run_cmd "git add utils.py"
run_cmd "git status"

step "Unstage it (keep the changes in working directory)"
run_cmd "git restore --staged utils.py"
run_cmd "git status"
explain "File is modified but no longer staged"

run_cmd "git restore utils.py"
explain "Reverted the working directory change too"

pause

step "Make a commit, then UNDO it with reset --soft"
printf "# TODO: add cross-validation\n" >> train.py
run_cmd "git add train.py"
run_cmd 'git commit -m "Add TODO comment (we will undo this)"'
run_cmd "git log --oneline -3"

step "Undo the commit but KEEP the changes staged"
run_cmd "git reset --soft HEAD~1"
run_cmd "git status"
explain "The commit is gone, but the change is still staged"
run_cmd "git log --oneline -3"

run_cmd "git restore --staged train.py"
run_cmd "git restore train.py"
explain "Cleaned up completely"

pause

step "Git stash: save work for later"
printf "# Work in progress: new feature\n" >> train.py
printf "experimental = True\n" >> config.yaml
run_cmd "git status"

step "Stash the changes (puts them in a pocket)"
run_cmd "git stash"
run_cmd "git status"
explain "Working directory is clean! Changes are safely stashed."

step "Get the stashed changes back"
run_cmd "git stash pop"
run_cmd "git status"
explain "Changes are restored. Stash and pop are your best friends."

run_silent "git restore train.py config.yaml"

pause

# =============================================================================
# SECTION 7: .gitignore
# =============================================================================

slide_ref "Advance to slide 21 — 'Demo §7: .gitignore'"
section "7. .GITIGNORE"

step "Create files that should NOT be tracked"
mkdir -p __pycache__ data
printf "cached bytecode" > __pycache__/train.cpython-310.pyc
printf "SECRET_API_KEY=sk-abc123" > .env
dd if=/dev/zero bs=1024 count=100 of=model.pkl 2>/dev/null
printf "1,2,3" > data/train.csv
explain "Created __pycache__/, .env, model.pkl, data/train.csv"
printf "\n"

step "Without .gitignore, Git sees everything"
run_cmd "git status"
explain "Git wants to track secrets, cache files, and large model files!"

pause

step "Create .gitignore with ML-appropriate rules"
explain "Create .gitignore with these contents:"
printf "${DIM}────────────────────────────────────────${NC}\n"
cat << 'SHOW'
# Python bytecode
__pycache__/
*.pyc

# Secrets -- NEVER commit these!
.env

# Large model files
*.pkl
*.h5

# Data files
data/*.csv
data/*.parquet

# IDE
.vscode/
.idea/

# OS
.DS_Store
SHOW
printf "${DIM}────────────────────────────────────────${NC}\n"

cat > .gitignore << 'EOF'
# Python bytecode
__pycache__/
*.pyc

# Secrets -- NEVER commit these!
.env

# Large model files
*.pkl
*.h5

# Data files
data/*.csv
data/*.parquet

# IDE
.vscode/
.idea/

# OS
.DS_Store
EOF

pause

step "Now check status — ignored files are hidden!"
run_cmd "git status"
explain "Only .gitignore itself shows up. Everything else is ignored."

run_cmd "git add .gitignore"
run_cmd 'git commit -m "Add .gitignore for ML project"'

show_state

pause

# =============================================================================
# SECTION 8: Simulating Remote (Push/Pull)
# =============================================================================

slide_ref "Advance to slide 23 — 'Demo §8: Push, Clone & Pull'"
section "8. SIMULATING A REMOTE (Push/Pull)"

step "Create a bare repo to act as 'GitHub'"
REMOTE_DIR="$DEMO_DIR/fake-github.git"
run_cmd "git init --bare $REMOTE_DIR"

step "Add it as a remote named 'origin'"
run_cmd "git remote add origin $REMOTE_DIR"
run_cmd "git remote -v"

pause

step "Push our work to the 'remote'"
run_cmd "git push -u origin main"
explain "All our commits are now on the 'remote' (like GitHub)"

pause

step "Simulate a collaborator: clone the repo elsewhere"
COLLAB_DIR="$DEMO_DIR/collaborator-clone"
run_cmd "git clone $REMOTE_DIR $COLLAB_DIR"

step "Collaborator makes a change and pushes"
cd "$COLLAB_DIR"
run_silent 'git config user.name "Collaborator"'
run_silent 'git config user.email "collab@iitgn.ac.in"'
run_silent 'git checkout main 2>/dev/null || git checkout -b main origin/main 2>/dev/null || true'
printf "\n## Setup\n\`\`\`bash\npip install -r requirements.txt\n\`\`\`\n" >> README.md
run_silent 'git add README.md'
run_silent 'git commit -m "Add setup instructions to README"'
run_silent 'git push origin main'
explain "Collaborator pushed a change to the remote"

pause

step "Back in our repo: pull the collaborator's change"
cd "$DEMO_DIR/ml-project"
run_cmd "git pull origin main"

step "We have their change!"
run_cmd "tail -5 README.md"
show_state

pause

# =============================================================================
# FINAL SUMMARY
# =============================================================================

section "DEMO COMPLETE!"

printf "${BOLD}Repository location:${NC} %s/ml-project\n" "$DEMO_DIR"
printf "${BOLD}Remote location:${NC}    %s\n" "$REMOTE_DIR"
printf "${BOLD}Clone location:${NC}     %s\n" "$COLLAB_DIR"
printf "\n"
printf "${GREEN}You can explore these directories to review what we did.${NC}\n"
printf "\n"
printf "${BOLD}Commands covered:${NC}\n"
printf "  git init, add, commit, status, diff, log\n"
printf "  git branch, checkout, merge\n"
printf "  git restore, reset, stash\n"
printf "  git remote, push, pull, clone\n"
printf "  .gitignore\n"
printf "\n"

git log --oneline --graph --all --decorate

printf "\n${BOLD}${GREEN}Happy Git-ing!${NC}\n"

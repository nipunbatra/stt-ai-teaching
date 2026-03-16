---
marp: true
theme: iitgn-modern
paginate: true
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Git Deep Dive

## Week 9: CS 203 - Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

Open **`git_followalong.sh`** in your editor + a terminal side by side.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Phase 0: The Problem

*Why do we need version control?*

---

# The Dark Ages

<img src="images/week08/version_control_chaos.png" width="500" style="display: block; margin: 0 auto;">

```
model.py
model_v2.py
model_v2_FINAL.py
model_v2_FINAL_actually_final.py
model_v2_FINAL_actually_final_nipun_edits.py
```

We've all been this person. Five copies, no idea which is real. **Let's fix this.**

---

# Enter Git: A Time Machine + Multiverse Engine

| Without Git | With Git |
|:-:|:-:|
| 5 files named `*_FINAL_v3` | One file, full history |
| "Which version had that feature?" | `git log` — every change, ever |
| "I broke everything!" | `git checkout` — go back in time |
| "Let me try something risky..." | `git branch` — a parallel universe |

Git gives you two superpowers:

1. **Time Machine** — travel to any point in your project's past
2. **Multiverse Engine** — experiment in parallel realities without risk

*Act 1: Let's feel the pain first.*

*⌨ Switch to terminal — Act 1: `mkdir ml-chaos`*

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Phase 1: The Personal Time Machine

*Acts 2 & 3 — Your first save points*

---

# The Three Areas of Git

<img src="images/week08/git_three_areas.png" width="800" style="display: block; margin: 0 auto;">

| Working Directory | Staging Area | Repository |
|:-:|:-:|:-:|
| Your Desk | The Shopping Cart | The Vault |
| You edit files here | `git add` puts things here | `git commit` locks them in |

You don't save *everything* at once — you **pick** what goes into each save point.

---

# The Letter Analogy

<img src="images/week08/git_letter_analogy.png" width="750" style="display: block; margin: 0 auto;">

| Step | Analogy | Git Command |
|:--:|:--|:--|
| 1 | Write the letter | Edit your files |
| 2 | Put it in the envelope | `git add <files>` |
| 3 | Drop it in the mailbox | `git commit -m "why"` |

The envelope (staging area) lets you **choose** which changes to include in this commit.

---

# Commits = Video Game Save Points

Think of commits like **save points in a video game**:

- You save **before the boss fight** (before a risky change)
- You save **after clearing a level** (after a feature works)
- If you die, you **load the last save** (not start the whole game over)

```
Save 1: "Add data loading"          ← level 1 clear
Save 2: "Add preprocessing"         ← level 2 clear
Save 3: "Try new model architecture" ← boss fight attempt
Save 4: "Fix model bug"             ← boss defeated!
```

**Commit message rule:** *"If applied, this commit will \_\_\_."*

Good: `"Add data augmentation pipeline"`
Bad: `"stuff"`, `"fix"`, `"asdfgh"`

---

# ⌨ Demo Break: Acts 2 & 3

**Act 2**: Create a project, initialize Git

```
mkdir ml-project && cd ml-project && git init
```

**Act 3**: The core rhythm

```
edit → git status → git diff → git add → git commit
```

*⌨ Switch to terminal — Acts 2 & 3*

---

# .gitignore — The Bouncer at the Door

Some files should **never** enter the vault:

| Do NOT Track | Why |
|:--|:--|
| `__pycache__/`, `*.pyc` | Generated files — anyone can recreate them |
| `.env`, `secrets.json` | Passwords and API keys — security risk! |
| `data/`, `*.csv`, `*.h5` | Large datasets — use DVC or cloud storage |
| `.DS_Store`, `Thumbs.db` | OS junk — nobody needs these |
| `wandb/`, `mlruns/` | Experiment logs — tracked separately |

```gitignore
# .gitignore — Git will pretend these files don't exist
__pycache__/
*.pyc
.env
data/
*.csv
```

Create this file **first thing** in every project.

---

# Time Travel: Loading Old Saves

| What You Want | Command | Game Analogy |
|:--|:--|:--|
| See save history | `git log --oneline` | List of save slots |
| What changed since last save? | `git diff` | Compare current to save |
| Load an old save (read-only) | `git checkout <hash>` | Watch a replay |
| Undo last save (keep changes) | `git reset --soft HEAD~1` | Re-do the boss fight |
| Discard file changes | `git restore <file>` | Reload from last save |

```
$ git log --oneline
a1b2c3d Add model training script
f4e5d6c Add data preprocessing
7g8h9i0 Initial commit — add README
```

Every commit has a unique ID (hash). You can always go back.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Phase 2: Parallel Universes

*Acts 4, 5, 6 — Branching*

---

# Branches: The "What If?" Machine

<img src="images/week08/git_parallel_universes.png" width="600" style="display: block; margin: 0 auto;">

Want to try a risky experiment? **Create a parallel universe.**

- Your working code stays perfectly safe on `main`
- Experiment wildly on `feature/new-model`
- If it works — merge it back
- If it fails — delete the branch, no harm done

---

# A Branch Is Just a Pointer

<img src="images/week08/git_branch_pointer.png" width="700" style="display: block; margin: 0 auto;">

A branch is **not** a copy of your files. It's just a sticky note pointing to a commit.

| Fact | Why It Matters |
|:--|:--|
| Creating a branch = writing a 40-char string | **Instant.** No copies made. |
| Switching branches swaps the sticky note | **Free.** Takes milliseconds. |
| You can have 100 branches | **Cheap.** No disk space wasted. |

```bash
git checkout -b feature/augmentation   # create + switch
git checkout main                      # switch back
```

---

# ⌨ Demo Break: Acts 4, 5, 6

**Act 4**: Create branches, switch between them

**Act 5**: Break things, undo them with reset and restore

**Act 6**: Work on a feature branch

*⌨ Switch to terminal — Acts 4, 5, 6*

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Phase 3: The Collision

*Acts 7 & 8 — Merging and conflicts*

---

# Merging: Stitching Timelines Together

<img src="images/week08/git_diverging_branches.png" width="650" style="display: block; margin: 0 auto;">

Both branches have new commits. Git creates a **merge commit** with two parents — stitching the timelines together.

```bash
git checkout main
git merge feature/augmentation
```

If the branches edited **different files** (or different parts of the same file), Git merges automatically. Easy.

But what if both branches edited **the same line**?

---

# Merge Conflicts: Two People, One Line

When two branches change the same line, Git cannot decide which version to keep. It marks the conflict:

```python
def learning_rate():
<<<<<<< HEAD
    return 0.001    # main branch says 0.001
=======
    return 0.01     # feature branch says 0.01
>>>>>>> feature/augmentation
```

| Marker | Meaning |
|:--|:--|
| `<<<<<<< HEAD` | Start of YOUR branch's version |
| `=======` | Divider |
| `>>>>>>> feature/...` | Start of the OTHER branch's version |

**Git is not broken.** It just needs you to decide.

---

# Resolving Conflicts: You Are the Arbiter

**Step by step:**

1. Open the conflicted file
2. Find the `<<<<<<<` markers
3. **Choose** which version to keep (or combine both)
4. Delete the markers
5. Stage and commit

```python
# After resolving — you picked 0.001 and added a comment:
def learning_rate():
    return 0.001    # validated in experiment #42
```

```bash
git add model.py
git commit -m "Resolve learning rate conflict — use validated value"
```

**The golden rule of conflicts:** read both versions, understand the intent, then decide.

---

# ⌨ Demo Break: Acts 7 & 8

**Act 7**: Merge a feature branch into main

**Act 8**: Create a conflict, resolve it

*⌨ Switch to terminal — Acts 7 & 8*

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Phase 4: The Pause Button + The Cloud

*Acts 9 & 10 — Stash and remotes*

---

# git stash: Shoving Mess Under the Bed

You're halfway through a feature. Your teammate says *"quick, check the main branch!"*

**Problem:** you can't switch branches with uncommitted changes.
**Solution:** stash them — shove the mess under the bed.

```bash
git stash              # hide all uncommitted changes
git checkout main      # do whatever you need
git checkout feature   # come back
git stash pop          # pull the mess back out
```

| Command | What It Does |
|:--|:--|
| `git stash` | Save WIP changes, clean working directory |
| `git stash list` | See all stashed changes |
| `git stash pop` | Restore the most recent stash |
| `git stash drop` | Throw away a stash |

---

# Git vs GitHub: Camera vs YouTube

<img src="images/week08/git_local_remote.png" width="750" style="display: block; margin: 0 auto;">

| Git | GitHub |
|:-:|:-:|
| The **camera** | **YouTube** |
| Records and stores locally | Shares with the world |
| Works offline | Needs internet |
| Free, open-source tool | A hosting service (one of many) |

Everything is **local** until you explicitly push. Git works perfectly without GitHub.

---

# push & pull: The Teleporter

Only **three commands** talk to the remote:

| Command | Direction | Analogy |
|:--|:-:|:--|
| `git push` | Local → Remote | Upload your saves to the cloud |
| `git pull` | Remote → Local | Download teammate's saves |
| `git clone` | Remote → Local | First-time full download |

```bash
git remote add origin https://github.com/user/repo.git
git push -u origin main      # first push: link local to remote
git push                     # subsequent pushes
git pull                     # get latest changes
```

**Think of it as syncing your game saves to the cloud** — so you can play on another machine (or your teammate can continue where you left off).

---

# ⌨ Demo Break: Acts 9 & 10

**Act 9**: Stash changes, switch branches, pop stash

**Act 10**: Push to GitHub, clone, pull

*⌨ Switch to terminal — Acts 9 & 10*

---

# Quick Reference

| Task | Command |
|------|---------|
| What changed? | `git status` / `git diff` |
| Stage files | `git add <files>` |
| Save checkpoint | `git commit -m "why"` |
| Visual history | `git log --oneline --graph` |
| New branch | `git checkout -b <name>` |
| Switch branch | `git checkout <name>` |
| Merge branch | `git merge <branch>` |
| Discard file changes | `git restore <file>` |
| Undo last commit (keep changes) | `git reset --soft HEAD~1` |
| Hide WIP | `git stash` / `git stash pop` |
| Upload to remote | `git push` |
| Download from remote | `git pull` |
| Ignore files | Edit `.gitignore` |

---

<!-- _class: lead -->
<!-- _paginate: false -->

# The Golden Rule

As long as you **commit**, you can almost always get your work back.

Uncommitted work is the only work Git cannot save.

**Commit early. Commit often. Write good messages.**

> *Next week: Version your ENVIRONMENT (venv, Docker)*

# Rendering all slides and website.

## 1. Prerequisites (WSL Setup)

1.  **Install WSL**: If you haven't already, open PowerShell as Administrator and run:
    ```powershell
    wsl --install
    ```
    (Restart your computer if prompted).

2.  **Open your project in WSL**:
    - Open your terminal (Ubuntu/WSL).
    - Navigate to your project folder (e.g., `cd /mnt/c/Users/YourName/path/to/repo`).

## 2. Install Dependencies (Inside WSL)

Run the following commands inside your WSL terminal to install the necessary tools:

```bash
# 1. Update package lists
sudo apt-get update

# 2. Install System Tools (Graphviz for diagrams, Python, Node.js)
sudo apt-get install -y graphviz python3 python3-pip nodejs npm

# 3. Install Quarto (Required for website)
# Download the latest .deb package (check quarto.org for newer versions if needed)
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.555/quarto-1.4.555-linux-amd64.deb
sudo dpkg -i quarto-1.4.555-linux-amd64.deb
rm quarto-1.4.555-linux-amd64.deb

# 4. Install Project Dependencies
npm install                          # Installs Marp CLI
pip3 install -r diagrams/requirements.txt  # Installs Python diagram libraries
```

## 3. Build Commands

### Building Slides (Using Make)
The `Makefile` automates the slide generation process.

```bash
# Build ALL slides (PDF and HTML)
make all

# Build slides for a specific week
make pdf/week01-data-collection-lecture.pdf

# Clean generated files
make clean

# List available slide sources
make list
```

### Building the Website
The website is built using Quarto (not handled by the Makefile).

```bash
quarto render
```

### Generating Diagrams
If you edit the Python scripts in `diagram-generators/`, regenerate the images:

```bash
# Generate a specific week's diagrams
python3 diagram-generators/week01_diagrams.py
```

## 4. Troubleshooting

### Error: "No suitable browser found" or "Snap" errors

If you see "No suitable browser found" or encounter errors installing `chromium-browser` (which tries to use Snap):



**Fix:** Snaps often fail in WSL. If you already tried installing `chromium-browser` and it failed:

```bash

sudo dpkg --remove --force-all chromium-browser

sudo apt-get install -f

```



Then, install Google Chrome directly (this works best):

```bash

wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

sudo apt install -y ./google-chrome-stable_current_amd64.deb

rm google-chrome-stable_current_amd64.deb

```

This provides a stable browser for Marp to use for PDF generation.



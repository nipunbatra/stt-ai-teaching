.PHONY: all clean help html pdf handout list site

# Find all .qmd files (excluding index.qmd)
SLIDES := $(filter-out index.qmd, $(wildcard *.qmd))
HTML_FILES := $(SLIDES:.qmd=.html)
PDF_FILES := $(SLIDES:.qmd=.pdf)
HANDOUT_FILES := $(SLIDES:.qmd=-handout.pdf)

# Default target
all: site pdf handout
	@echo "✓ All slides built successfully"

# Build entire Quarto site (including index)
site:
	@echo "Building Quarto site..."
	@quarto render

# Build all HTML slides individually
html: $(HTML_FILES)
	@echo "✓ HTML slides built"

# Build all PDF slides (requires HTML first)
pdf: html $(PDF_FILES)
	@echo "✓ PDF slides built"

# Build all handout PDFs
handout: $(HANDOUT_FILES)
	@echo "✓ Handout PDFs built"

# Pattern rule for HTML
%.html: %.qmd
	@echo "Building $< -> $@"
	@quarto render $< --to revealjs

# Pattern rule for PDF slides (from RevealJS HTML using decktape)
%.pdf: %.html
	@echo "Converting $< -> $@"
	@npx --yes decktape reveal $< $@ --chrome-arg=--no-sandbox

# Pattern rule for handout PDF (using Beamer)
%-handout.pdf: %.qmd
	@echo "Building handout $< -> $@"
	@quarto render $< --to beamer --output $@

# Build specific slide by name (without extension) - all three formats
%: %.qmd
	@echo "Building $< (all formats)"
	@quarto render $< --to revealjs
	@npx --yes decktape reveal $*.html $*.pdf --chrome-arg=--no-sandbox
	@quarto render $< --to beamer --output $*-handout.pdf
	@echo "✓ Built $< (HTML + PDF slides + PDF handout)"

# List all available slides
list:
	@echo "Available slides:"
	@for file in $(SLIDES); do \
		echo "  - $${file%.qmd}"; \
	done
	@echo ""
	@echo "Usage:"
	@echo "  make all                  # Build all formats (HTML + PDF slides + PDF handouts)"
	@echo "  make html                 # Build all HTML slides"
	@echo "  make pdf                  # Build all PDF slides (decktape)"
	@echo "  make handout              # Build all PDF handouts (beamer)"
	@echo "  make <name>               # Build specific slide in all formats"
	@echo "  make clean                # Remove generated files"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f $(HTML_FILES) $(PDF_FILES) $(HANDOUT_FILES) index.html
	@rm -rf *_files/
	@rm -f *.tex *.log *.aux
	@rm -rf .quarto/
	@echo "✓ Clean complete"

# Help
help: list

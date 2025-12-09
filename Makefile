.PHONY: all clean help list dirs

# Directories
SLIDES_DIR := slides
PDF_DIR := pdf
HTML_DIR := html

# Find all .md files in slides/ directory
SLIDES_MD := $(wildcard $(SLIDES_DIR)/*.md)

# Define output files
# Map slides/%.md -> pdf/%.pdf
PDF_TARGETS := $(patsubst $(SLIDES_DIR)/%.md, $(PDF_DIR)/%.pdf, $(SLIDES_MD))
# Map slides/%.md -> html/%.html
HTML_TARGETS := $(patsubst $(SLIDES_DIR)/%.md, $(HTML_DIR)/%.html, $(SLIDES_MD))

# Default target
all: dirs $(PDF_TARGETS) $(HTML_TARGETS)
	@echo "✓ All slides built successfully"

# Create output directories
dirs:
	@mkdir -p $(PDF_DIR)
	@mkdir -p $(HTML_DIR)

# Pattern rule for PDF (Note: Mermaid diagrams will appear as code blocks in PDF)
$(PDF_DIR)/%.pdf: $(SLIDES_DIR)/%.md
	@echo "Building PDF: $< -> $@"
	@marp $< -o $@ --pdf --allow-local-files

# Pattern rule for HTML
$(HTML_DIR)/%.html: $(SLIDES_DIR)/%.md
	@echo "Building HTML: $< -> $@"
	@marp $< -o $@ --html --allow-local-files

# List available slides
list:
	@echo "Available slides:"
	@for file in $(SLIDES_MD); do \
		echo "  - $$file"; \
	done

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf $(PDF_DIR) $(HTML_DIR)
	@echo "✓ Clean complete"

help:
	@echo "Available targets:"
	@echo "  make all       # Build all slides (PDF and HTML) - default"
	@echo "  make clean     # Remove generated pdf/ and html/ directories"
	@echo "  make list      # List available source slides"
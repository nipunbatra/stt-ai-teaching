.PHONY: all clean help list dirs

# Directories
SLIDES_DIR := slides
PDF_DIR := pdf
HTML_DIR := html

# Theme
THEME := $(SLIDES_DIR)/iitgn-modern.css

# Find all .md files in slides/ directory
SLIDES_MD := $(wildcard $(SLIDES_DIR)/*.md)

# Extract week names (e.g., week01, week02, week04)
WEEKS := $(sort $(patsubst $(SLIDES_DIR)/%-lecture.md,%,$(filter %-lecture.md,$(SLIDES_MD))))

# Define output files
PDF_TARGETS := $(patsubst $(SLIDES_DIR)/%.md, $(PDF_DIR)/%.pdf, $(SLIDES_MD))
HTML_TARGETS := $(patsubst $(SLIDES_DIR)/%.md, $(HTML_DIR)/%.html, $(SLIDES_MD))

# Default target
all: dirs $(PDF_TARGETS) $(HTML_TARGETS)
	@echo "✓ All slides built successfully"

# Create output directories and copy images
dirs:
	@mkdir -p $(PDF_DIR)
	@mkdir -p $(HTML_DIR)
	@if [ -d "$(SLIDES_DIR)/images" ]; then \
		cp -r $(SLIDES_DIR)/images $(HTML_DIR)/; \
	fi

# Pattern rule for PDF
$(PDF_DIR)/%.pdf: $(SLIDES_DIR)/%.md $(THEME) | dirs
	@echo "Building PDF: $< -> $@"
	@npx marp $< -o $@ --pdf --allow-local-files --theme-set $(THEME)

# Pattern rule for HTML
$(HTML_DIR)/%.html: $(SLIDES_DIR)/%.md $(THEME) | dirs
	@echo "Building HTML: $< -> $@"
	@npx marp $< -o $@ --html --allow-local-files --theme-set $(THEME)

# HTML only for a week (faster) - must come before week% rule
week%-html: dirs
	@echo "Building week$* HTML only..."
	@for f in $(SLIDES_DIR)/week$*-*.md; do \
		if [ -f "$$f" ]; then \
			name=$$(basename "$$f" .md); \
			echo "  $$f -> $(HTML_DIR)/$$name.html"; \
			npx marp "$$f" -o "$(HTML_DIR)/$$name.html" --html --allow-local-files --theme-set $(THEME); \
		fi \
	done
	@echo "✓ week$* HTML done"

# Build specific week (e.g., make week04)
week%: dirs
	@echo "Building week$* slides..."
	@for f in $(SLIDES_DIR)/week$*-*.md; do \
		if [ -f "$$f" ]; then \
			name=$$(basename "$$f" .md); \
			echo "  HTML: $$f -> $(HTML_DIR)/$$name.html"; \
			npx marp "$$f" -o "$(HTML_DIR)/$$name.html" --html --allow-local-files --theme-set $(THEME); \
			echo "  PDF:  $$f -> $(PDF_DIR)/$$name.pdf"; \
			npx marp "$$f" -o "$(PDF_DIR)/$$name.pdf" --pdf --allow-local-files --theme-set $(THEME) 2>/dev/null || echo "  (PDF skipped - needs Chrome)"; \
		fi \
	done
	@echo "✓ week$* done"

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
	@echo "Usage:"
	@echo "  make week04       # Build week04 HTML + PDF"
	@echo "  make week04-html  # Build week04 HTML only (fast)"
	@echo "  make all          # Build everything"
	@echo "  make list         # List available slides"
	@echo "  make clean        # Remove generated files"
	@echo ""
	@echo "Available weeks: $(WEEKS)"
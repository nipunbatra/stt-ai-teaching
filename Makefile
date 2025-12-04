.PHONY: all clean help pdf list light dark

# Find all .typ files (excluding slides.typ and other partials)
SLIDES := $(filter-out slides.typ, $(wildcard *.typ))
PDF_LIGHT := $(SLIDES:.typ=-light.pdf)
PDF_DARK := $(SLIDES:.typ=-dark.pdf)

# Default target - build light version
all: light
	@echo "✓ All slides built successfully"

# Build light (white background) PDFs
light: $(PDF_LIGHT)
	@echo "✓ Light theme PDFs built"

# Build dark PDFs
dark: $(PDF_DARK)
	@echo "✓ Dark theme PDFs built"

# Build both versions
both: light dark
	@echo "✓ Both light and dark PDFs built"

# Pattern rule for light PDF
%-light.pdf: %.typ slides.typ
	@echo "Compiling $< -> $@ (light theme)"
	@typst compile $< $@ --input dark-mode=false

# Pattern rule for dark PDF
%-dark.pdf: %.typ slides.typ
	@echo "Compiling $< -> $@ (dark theme)"
	@typst compile $< $@ --input dark-mode=true

# For compatibility: build main PDF as light version
%.pdf: %.typ slides.typ
	@echo "Compiling $< -> $@"
	@typst compile $< $@ --input dark-mode=false

# List available slides
list:
	@echo "Available slides:"
	@for file in $(SLIDES); do \
		echo "  - $${file%.typ}"; \
	done
	@echo ""
	@echo "Build commands:"
	@echo "  make light     # Build light theme PDFs (white background)"
	@echo "  make dark      # Build dark theme PDFs (dark background)"
	@echo "  make both      # Build both themes"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f $(PDF_LIGHT) $(PDF_DARK) *.pdf
	@echo "✓ Clean complete"

help:
	@echo "Available targets:"
	@echo "  make light     # Build light theme PDFs (default)"
	@echo "  make dark      # Build dark theme PDFs"
	@echo "  make both      # Build both light and dark themes"
	@echo "  make list      # List available slides"
	@echo "  make clean     # Remove all generated PDFs"

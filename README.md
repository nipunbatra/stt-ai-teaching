# CS 203: Software Tools and Techniques for AI

Course materials for CS 203 at IIT Gandhinagar.

Course website: https://nipunbatra.github.io/stt-ai-26/

## Slides

### Data Collection and Labeling

Comprehensive slides covering:
- Data Collection (instrumentation, analytics, logging, scraping, streaming)
- Data Validation (Pydantic, Great Expectations, Pandera, quality monitoring)
- Data Labeling (Label Studio, inter-annotator agreement, active learning, weak supervision)
- Data Augmentation (image, text, audio, time series, SMOTE, generative models)

**Files:**
- Source: `data-collection-labeling.qmd`
- PDF: `data-collection-labeling.pdf`

## Building Slides

### Prerequisites

Install Quarto: https://quarto.org/docs/get-started/

### Render HTML Slides

```bash
quarto render data-collection-labeling.qmd --to revealjs
```

This creates `data-collection-labeling.html` - open in browser to present.

**Navigation:**
- Arrow keys / Space: Next slide
- `F`: Fullscreen
- `S`: Speaker notes
- `C`: Chalkboard (draw on slides)

### Export to PDF (Document Style)

```bash
quarto render data-collection-labeling.qmd --to pdf
```

## Editing Slides

Edit the `.qmd` file directly - it's Markdown with special syntax for slides.

**Slide breaks:**
- `#` - New section (title slide)
- `##` - New slide

**Incremental lists:**
```markdown
::: incremental
- Item 1
- Item 2
:::
```

**Code blocks:**
````markdown
```python
# Your code here
```
````

**Columns:**
```markdown
::: columns
::: {.column width="50%"}
Left content
:::
::: {.column width="50%"}
Right content
:::
:::
```

## License

Course materials by Prof. Nipun Batra, IIT Gandhinagar.

#import "@preview/touying:0.5.3": *
#import themes.dewdrop: *

// Clean, minimal theme with light/dark mode support
#let course-theme(
  title: "",
  subtitle: "",
  author: "",
  date: datetime.today(),
  dark-mode: false,
) = {
  // Color scheme
  let colors = if dark-mode {
    (
      bg: rgb("#0f0f23"),
      fg: rgb("#e2e8f0"),
      accent: rgb("#60a5fa"),
      muted: rgb("#94a3b8"),
      code-bg: rgb("#1a1b26"),
      code-border: rgb("#2a2b36"),
      card-bg: rgb("#1e293b"),
    )
  } else {
    (
      bg: rgb("#ffffff"),
      fg: rgb("#1e293b"),
      accent: rgb("#2563eb"),
      muted: rgb("#64748b"),
      code-bg: rgb("#f8fafc"),
      code-border: rgb("#e2e8f0"),
      card-bg: rgb("#f1f5f9"),
    )
  }

  let theme = dewdrop-theme.with(
    aspect-ratio: "16-9",
    navigation: "none",
    config-info(
      title: title,
      subtitle: subtitle,
      author: author,
      date: date,
      institution: [IIT Gandhinagar],
    ),
    config-colors(
      primary: colors.accent,
      secondary: colors.fg,
      tertiary: colors.muted,
      neutral-lightest: colors.bg,
      neutral-darkest: colors.fg,
    ),
  )

  body => {
    set page(fill: colors.bg)

    set text(
      font: ("Inter", "SF Pro Display", "Helvetica Neue"),
      size: 22pt,
      fill: colors.fg,
    )

    // Headings
    show heading.where(level: 1): it => {
      set text(size: 42pt, weight: "bold", fill: colors.fg)
      block(spacing: 0.8em, it)
    }

    show heading.where(level: 2): it => {
      set text(size: 28pt, weight: "semibold", fill: colors.accent)
      block(spacing: 0.6em, it)
    }

    show heading.where(level: 3): it => {
      set text(size: 24pt, weight: "semibold", fill: colors.fg)
      block(spacing: 0.5em, it)
    }

    // Code blocks with syntax highlighting
    show raw.where(block: true): it => {
      block(
        fill: colors.code-bg,
        stroke: 1pt + colors.code-border,
        inset: 1em,
        radius: 8pt,
        width: 100%,
        text(fill: colors.fg, size: 18pt, it)
      )
    }

    // Inline code
    show raw.where(block: false): box.with(
      fill: colors.code-bg,
      inset: (x: 5pt, y: 2pt),
      outset: (y: 2pt),
      radius: 3pt,
    )

    // Lists
    set list(marker: text(fill: colors.accent, "•"))
    set enum(numbering: n => text(fill: colors.accent, weight: "bold", str(n) + "."))

    // Links
    show link: set text(fill: colors.accent)

    // Strong text
    show strong: set text(fill: colors.accent, weight: "semibold")

    theme(body)
  }
}

// Title slide
#let title-slide(title, subtitle: none, dark-mode: false) = {
  let gradient-colors = if dark-mode {
    (rgb("#1e3a8a"), rgb("#0f172a"))
  } else {
    (rgb("#3b82f6"), rgb("#1e40af"))
  }

  set page(
    fill: gradient.linear(
      gradient-colors.at(0),
      gradient-colors.at(1),
      angle: 135deg,
    )
  )
  align(center + horizon)[
    #text(size: 52pt, weight: "bold", fill: white)[#title]
    #if subtitle != none {
      v(0.5em)
      text(size: 26pt, fill: rgb("#cbd5e1"))[#subtitle]
    }
  ]
}

// Section divider
#let section-slide(title, dark-mode: false) = {
  let gradient-colors = if dark-mode {
    (rgb("#3b82f6"), rgb("#1e40af"))
  } else {
    (rgb("#60a5fa"), rgb("#2563eb"))
  }

  set page(
    fill: gradient.linear(
      gradient-colors.at(0),
      gradient-colors.at(1),
      angle: 45deg,
    )
  )
  align(center + horizon)[
    #text(size: 48pt, weight: "bold", fill: white)[#title]
  ]
}

// Two columns
#let columns-layout(left, right) = {
  grid(
    columns: (1fr, 1fr),
    gutter: 2.5em,
    left,
    right
  )
}

// Callout boxes
#let tip-box(body, dark-mode: false) = {
  let colors = if dark-mode {
    (bg: rgb("#1e3a5f"), border: rgb("#3b82f6"), text: rgb("#93c5fd"))
  } else {
    (bg: rgb("#eff6ff"), border: rgb("#3b82f6"), text: rgb("#1e40af"))
  }

  block(
    fill: colors.bg,
    stroke: 2pt + colors.border,
    inset: 1em,
    radius: 8pt,
    width: 100%,
  )[
    #text(fill: colors.text, weight: "bold")[💡 Tip] \
    #body
  ]
}

#let warning-box(body, dark-mode: false) = {
  let colors = if dark-mode {
    (bg: rgb("#422006"), border: rgb("#f59e0b"), text: rgb("#fcd34d"))
  } else {
    (bg: rgb("#fffbeb"), border: rgb("#f59e0b"), text: rgb("#92400e"))
  }

  block(
    fill: colors.bg,
    stroke: 2pt + colors.border,
    inset: 1em,
    radius: 8pt,
    width: 100%,
  )[
    #text(fill: colors.text, weight: "bold")[⚠️ Warning] \
    #body
  ]
}

#let info-box(body, dark-mode: false) = {
  let colors = if dark-mode {
    (bg: rgb("#0f172a"), border: rgb("#60a5fa"), text: rgb("#93c5fd"))
  } else {
    (bg: rgb("#f0f9ff"), border: rgb("#0ea5e9"), text: rgb("#0c4a6e"))
  }

  block(
    fill: colors.bg,
    stroke: 2pt + colors.border,
    inset: 1em,
    radius: 8pt,
    width: 100%,
  )[
    #text(fill: colors.text, weight: "bold")[ℹ️ Note] \
    #body
  ]
}

#let card(body, dark-mode: false) = {
  let bg = if dark-mode { rgb("#1e293b") } else { rgb("#f8fafc") }

  block(
    fill: bg,
    inset: 1.2em,
    radius: 10pt,
    width: 100%,
    body
  )
}

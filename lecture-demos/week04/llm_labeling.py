"""
LLM Labeling Demo
=================
This demo shows how to use OpenAI/Anthropic APIs to label data.

Run: python llm_labeling.py

Requirements: pip install openai anthropic
"""

import os

# Choose your provider
USE_OPENAI = True  # Set to False to use Anthropic

# Sample movie reviews to label
reviews = [
    "Mind-blowing visuals! Nolan does it again!",
    "Meh. Seen better, seen worse.",
    "Two hours of my life I'll never get back.",
    "A masterpiece of modern cinema!",
    "Boring plot, terrible acting.",
    "It was okay, nothing special.",
]

# =============================================================================
# OpenAI Implementation
# =============================================================================

def label_with_openai(review):
    """Label a single review using OpenAI GPT."""
    from openai import OpenAI
    client = OpenAI()  # Uses OPENAI_API_KEY env var

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use gpt-4 for better quality
        messages=[
            {
                "role": "system",
                "content": """Classify movie reviews as POSITIVE, NEGATIVE, or NEUTRAL.

POSITIVE: Reviewer enjoyed/recommends the movie
NEGATIVE: Reviewer disliked/doesn't recommend the movie
NEUTRAL: Mixed feelings or no strong opinion

Respond with ONLY the label, nothing else."""
            },
            {
                "role": "user",
                "content": f"Review: {review}"
            }
        ],
        max_tokens=10,
        temperature=0  # Deterministic output
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# Anthropic Implementation
# =============================================================================

def label_with_anthropic(review):
    """Label a single review using Anthropic Claude."""
    import anthropic
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    response = client.messages.create(
        model="claude-3-haiku-20240307",  # Fast and cheap
        max_tokens=10,
        system="""Classify movie reviews as POSITIVE, NEGATIVE, or NEUTRAL.

POSITIVE: Reviewer enjoyed/recommends the movie
NEGATIVE: Reviewer disliked/doesn't recommend the movie
NEUTRAL: Mixed feelings or no strong opinion

Respond with ONLY the label, nothing else.""",
        messages=[
            {"role": "user", "content": f"Review: {review}"}
        ]
    )
    return response.content[0].text.strip()


# =============================================================================
# Few-Shot Prompting (Better Accuracy)
# =============================================================================

def label_with_few_shot(review, label_func):
    """Use few-shot examples for better accuracy."""
    few_shot_prompt = """Classify movie review sentiment.

Examples:
Review: "Absolutely loved it! Best movie of the year!"
Label: POSITIVE

Review: "Waste of time. Don't bother watching."
Label: NEGATIVE

Review: "It was fine. Nothing special but not bad."
Label: NEUTRAL

Review: "{review}"
Label:"""

    return label_func(few_shot_prompt.format(review=review))


# =============================================================================
# Batch Labeling
# =============================================================================

def batch_label(reviews, label_func):
    """Label multiple reviews."""
    results = []
    for i, review in enumerate(reviews):
        try:
            label = label_func(review)
            results.append({"review": review, "label": label})
            print(f"[{i+1}/{len(reviews)}] {label}: {review[:50]}...")
        except Exception as e:
            print(f"Error labeling review {i+1}: {e}")
            results.append({"review": review, "label": "ERROR"})
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LLM LABELING DEMO")
    print("=" * 60)

    # Check for API keys
    if USE_OPENAI:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nError: Set OPENAI_API_KEY environment variable")
            print("export OPENAI_API_KEY='your-key-here'")
            exit(1)
        label_func = label_with_openai
        print("\nUsing: OpenAI GPT-3.5-turbo")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\nError: Set ANTHROPIC_API_KEY environment variable")
            print("export ANTHROPIC_API_KEY='your-key-here'")
            exit(1)
        label_func = label_with_anthropic
        print("\nUsing: Anthropic Claude Haiku")

    print("\n" + "=" * 60)
    print("LABELING REVIEWS")
    print("=" * 60 + "\n")

    results = batch_label(reviews, label_func)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for r in results:
        print(f"  [{r['label']:8}] {r['review'][:50]}...")

    # Count labels
    from collections import Counter
    label_counts = Counter(r['label'] for r in results)
    print(f"\nLabel distribution: {dict(label_counts)}")

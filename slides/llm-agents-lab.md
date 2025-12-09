---
marp: true
theme: default
paginate: true
---

# Lab: LLM Agents

**Objective**: Build an autonomous research agent.

## Task 1: Tools Setup
- Get API key for Tavily (Search) or set up a mock search tool.
- Define a Python function `search(query: str)`.

## Task 2: Agent Loop
- Use LangGraph or raw Python loop.
- Prompt: "You have access to tools. Use them to answer."
- Parse LLM output -> Execute Tool -> Feed back result.

## Task 3: Complex Query
- Ask: "Compare the battery life of iPhone 15 vs Pixel 8."
- Agent should search for both, then synthesize.

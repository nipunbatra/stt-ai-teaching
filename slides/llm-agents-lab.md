---
marp: true
theme: default
paginate: true
style: |
  section { background: white; font-family: 'Inter', sans-serif; font-size: 28px; padding-top: 0; justify-content: flex-start; }
  h1 { color: #1e293b; border-bottom: 3px solid #f59e0b; font-size: 1.6em; margin-bottom: 0.5em; margin-top: 0; }
  h2 { color: #334155; font-size: 1.2em; margin: 0.5em 0; }
  code { background: #f8f9fa; font-size: 0.85em; font-family: 'Fira Code', monospace; border: 1px solid #e2e8f0; }
  pre { background: #f8f9fa; border-radius: 6px; padding: 1em; margin: 0.5em 0; }
  pre code { background: transparent; color: #1e293b; font-size: 0.7em; line-height: 1.5; }
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

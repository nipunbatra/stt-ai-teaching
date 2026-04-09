"""Generate Colab notebooks for Week 12: Building AI Agents from Scratch.

Run:
    python lecture-demos/week12/generate_colab_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "colab-notebooks"


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": filename},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = OUTDIR / filename
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote colab-notebooks/{filename}")


# ============================================================================
# 01 — Agents from Scratch with Gemma 4
# ============================================================================
def nb_agents() -> list[dict]:
    return [
        markdown_cell(
            """# Building AI Agents from Scratch with Gemma 4

> **Week 12 Lab** — CS 203: Software Tools and Techniques for AI

## What is an agent?

An **agent** = LLM + tools + a loop.

- **LLM** — the "brain" that reads your question, reasons about it, and decides what to do next.
- **Tools** — Python functions the LLM can ask you to call (a calculator, a weather API, a file reader, etc.).
- **Loop** — you keep asking the LLM "what next?" and executing tool calls until it says "I'm done, here's the answer."

**The LLM never runs any code.** It just *describes* what it wants (as a JSON object), and *your* code actually runs the function. The result goes back to the LLM, and it decides the next step.

> **Real-world examples:** Claude Code reads/edits files from your terminal. Google Assistant calls the clock API when you say "set a timer." Perplexity calls a search API and then summarizes the results. They're all the same pattern.

## What you will build

By the end of this notebook you will have:

1. Loaded **Gemma 4 E2B** (a 2B-parameter open model) on a free Colab T4 GPU
2. Made it answer a simple question (standard LLM usage)
3. Defined **four tools** (calculator, weather, unit converter, course notes)
4. Watched the model **choose** the right tool and use it
5. Built a **complete agent loop** that chains multiple tool calls together
6. **Added your own tool** and tested that the agent uses it

**Runtime:** Make sure you're on a **T4 GPU** runtime:
`Runtime → Change runtime type → T4 GPU`

**Time:** ~60–90 minutes
"""
        ),
        markdown_cell(
            """---

## Step 0: Install and import everything

We need two key packages:
- **`transformers`** — Hugging Face's library for loading and running LLMs
- **`bitsandbytes`** — enables 4-bit quantization so Gemma 4 fits on a free T4 GPU (remember Week 11!)

This cell takes ~1 minute.
"""
        ),
        code_cell(
            """!pip install -U transformers accelerate bitsandbytes -q
"""
        ),
        code_cell(
            """import json
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), "GB")
"""
        ),
        markdown_cell(
            """---

## Step 1: Load Gemma 4 E2B on a T4 GPU

### What is Gemma 4 E2B?

[**Gemma 4 E2B**](https://huggingface.co/google/gemma-4-e2b-it) is a 2-billion parameter open model from Google, released under the Apache 2.0 license. The **"E2B"** means it's part of the efficient model family, and **"it"** means it's instruction-tuned (trained to follow instructions and have conversations).

Most importantly for us: Gemma 4 has **native tool-calling support** — it was trained to output structured JSON tool calls, not just free-form text. This makes it ideal for building agents.

### Why 4-bit quantization?

At full precision (FP16), Gemma 4 E2B needs ~4 GB of VRAM. With **4-bit quantization** (using `bitsandbytes`), we cut that to ~1.5 GB — well within the free T4's 15 GB.

If you did Week 11, this is the same idea: represent each weight with fewer bits to shrink the model. We use the `nf4` (NormalFloat4) format with double quantization for the best quality-size tradeoff.

This cell takes ~60 seconds the first time (downloading ~1.5 GB) and ~15 seconds on subsequent runs (loading from cache).
"""
        ),
        code_cell(
            """MODEL_ID = "google/gemma-4-e2b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
print("Model loaded!")
"""
        ),
        markdown_cell(
            """---

## Step 2: Basic generation (no tools yet)

Before we add tools, let's build a helper function and confirm the model works.

### The `generate()` helper

This function wraps a lot of boilerplate into one call. Here's what each part does:

1. **`apply_chat_template`** — converts our Python list of messages (user, assistant, tool) into the specific token format Gemma 4 expects. When we pass `tools=...`, it also injects the tool descriptions into the prompt so the model knows what's available.
2. **`model.generate`** — runs the model and produces output tokens.
3. **Decode** — converts tokens back to text. We decode twice: once *with* special tokens (for parsing tool calls) and once *without* (for clean display).
4. **`parse_response`** — extracts structured tool calls from the raw output, if any. This is Gemma 4's built-in parser that turns the model's JSON output into a Python dict.
"""
        ),
        code_cell(
            """def generate(messages, tools=None, max_new_tokens=512):
    \"\"\"Helper: apply chat template, generate, decode, parse.\"\"\"
    kwargs = dict(
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    if tools:
        kwargs["tools"] = tools

    inputs = processor.apply_chat_template(messages, **kwargs).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    gen_ids = output[0][inputs.input_ids.shape[-1]:]
    raw = processor.decode(gen_ids, skip_special_tokens=False)
    text = processor.decode(gen_ids, skip_special_tokens=True)
    parsed = processor.parse_response(raw)
    return {"raw": raw, "text": text, "parsed": parsed}
"""
        ),
        markdown_cell(
            """### Test: a question the model can answer from memory
"""
        ),
        code_cell(
            """messages = [{"role": "user", "content": "What is the capital of India?"}]
result = generate(messages)
print(result["text"])
"""
        ),
        markdown_cell(
            """The model answered from memory — it knows facts from its training data. Now let's see what happens when we ask something it **can't** answer from memory:
"""
        ),
        code_cell(
            """messages = [{"role": "user", "content": "What is the weather in Gandhinagar right now?"}]
result = generate(messages)
print(result["text"])
"""
        ),
        markdown_cell(
            """The model probably said something like *"I don't have access to real-time data"* or it guessed based on general knowledge.

**This is the core problem we're going to solve.** An LLM is just a text predictor — it can only generate the next word. It can't open a browser, call an API, run code, or check a database.

But what if the model could say: *"I need to check the weather. Let me call a weather API."* — and then your code actually calls the API, gets the result, and feeds it back to the model?

**That's exactly what we're about to build.**

---

## Step 3: Your first tool — the calculator

### Why does an LLM need a calculator?

LLMs predict tokens — they don't compute. Ask a hard math question and they'll confidently give the wrong answer. Let's see:
"""
        ),
        code_cell(
            """messages = [{"role": "user", "content": "What is 4729 times 8314?"}]
result = generate(messages)
print("LLM says:", result["text"])
print("Actual  :", 4729 * 8314)
"""
        ),
        markdown_cell(
            """Probably wrong! LLMs are great at language but terrible at arithmetic. Let's fix this by giving the model a calculator tool.

### How function calling works

**The key insight: function calling is NOT the model running code.** The model never executes anything. Here's the flow:

| Step | Who does it | What happens |
|:--:|:--|:--|
| 1 | **You** | Hand the model a *menu* of available tools (JSON schemas) |
| 2 | **Model** | Reads the question, *decides* which tool to call |
| 3 | **Model** | Outputs a *structured request*: `{"name": "calculate", "args": {"expression": "4729 * 8314"}}` |
| 4 | **You** | Execute the function with those arguments, get the result |
| 5 | **You** | Feed the result *back* to the model |
| 6 | **Model** | Writes the final answer using the real data |

The model is a **decision maker**. You are the **executor**.

### Defining a tool

A tool has two parts:
1. A **Python function** that does the actual work (the model never sees this code)
2. A **JSON schema** that describes the function — name, what it does, what arguments it expects (this is what the model reads to decide)

Think of it like a restaurant menu: the diner (model) reads the menu (schema) and places an order. The kitchen (your code) actually makes the food (runs the function).
"""
        ),
        code_cell(
            """# Part 1: The actual function
def calculate(expression: str) -> str:
    \"\"\"Evaluate a math expression. Only allows numbers and basic operators.\"\"\"
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: only numbers and +-*/() are allowed"
    try:
        return str(round(eval(expression), 10))
    except Exception as e:
        return f"Error: {e}"


# Part 2: The schema (the "menu" we hand to the model)
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use Python syntax: + - * / ** () sqrt() etc. Examples: '4729 * 8314', '(50 + 30) / 4', '144 ** 0.5'",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 3 * 4'"
                }
            },
            "required": ["expression"]
        }
    }
}

# Quick test: does the function work?
print(calculate("4729 * 8314"))
print(calculate("144 ** 0.5"))
"""
        ),
        markdown_cell(
            """### Making the model use the calculator

Now we pass `tools=[calculator_tool]` when generating. Behind the scenes, `apply_chat_template` injects the tool's JSON schema into the prompt, so the model sees something like:

> *"You have the following tools available: calculate — Evaluate a mathematical expression..."*

The model reads this "menu" and **decides** whether to use the tool or answer directly.
"""
        ),
        code_cell(
            """messages = [{"role": "user", "content": "What is 4729 times 8314?"}]
result = generate(messages, tools=[calculator_tool])

print("=== Parsed response ===")
print(json.dumps(result["parsed"], indent=2))
"""
        ),
        markdown_cell(
            """**Look at the output carefully.** You should see a `tool_calls` list with `"name": "calculate"` and `"arguments": {"expression": "4729 * 8314"}`.

The model did NOT compute the answer — it just said *"please run this function with these arguments."* Now **we** execute the function and feed the result back:
"""
        ),
        code_cell(
            """# Extract the tool call
tool_call = result["parsed"]["tool_calls"][0]
print(f"Model wants to call: {tool_call['name']}({tool_call['arguments']})")

# Execute it
answer = calculate(**tool_call["arguments"])
print(f"Tool returned: {answer}")

# Feed the result back to the model
messages.append({"role": "assistant", "tool_calls": result["parsed"]["tool_calls"]})
messages.append({"role": "tool", "name": tool_call["name"], "content": answer})

# Generate the final answer
final = generate(messages, tools=[calculator_tool])
print(f"\\nFinal answer: {final['text']}")
"""
        ),
        markdown_cell(
            """The model now gives the **exact** answer: 39,317,006.

### What just happened — step by step

Let's slow down and really understand the message flow:

```
messages = [
  {"role": "user",      "content": "What is 4729 times 8314?"}         # ① You ask
  {"role": "assistant", "tool_calls": [{"name": "calculate", ...}]}    # ② Model requests tool
  {"role": "tool",      "name": "calculate", "content": "39317006"}    # ③ You execute, feed result
  # Now the model sees all three messages and writes a final answer     # ④ Model responds
]
```

This is the **entire mechanism** behind function calling. Every agent — Claude Code, ChatGPT with browsing, Google Assistant — works this way. The rest of this notebook is just making it more powerful.

---

## Step 4: Three more tools

Now let's build a toolkit of four tools. Each one addresses a different limitation of LLMs:

| Tool | LLM limitation | What the tool does |
|:--|:--|:--|
| `calculate` | LLMs can't do exact arithmetic | Evaluates math expressions |
| `get_weather` | LLMs have no real-time data | Looks up current weather |
| `convert_units` | LLMs approximate conversions | Precise unit conversion |
| `search_notes` | LLMs don't know *your* course | Searches CS 203 topics |

Together, these four tools will let the model answer questions it could never answer alone.

### Tool 2: Weather lookup
"""
        ),
        code_cell(
            """# --- Tool 2: Weather lookup (real-time data the model doesn't have) ---
def get_weather(city: str) -> str:
    \"\"\"Get current weather for a city (mock data for demo).\"\"\"
    data = {
        "gandhinagar": {"temp_c": 38, "condition": "Sunny",  "humidity": 25},
        "mumbai":      {"temp_c": 32, "condition": "Humid",  "humidity": 80},
        "bangalore":   {"temp_c": 28, "condition": "Rainy",  "humidity": 65},
        "delhi":       {"temp_c": 40, "condition": "Haze",   "humidity": 30},
        "chennai":     {"temp_c": 35, "condition": "Cloudy", "humidity": 70},
        "kolkata":     {"temp_c": 33, "condition": "Humid",  "humidity": 75},
        "paris":       {"temp_c": 18, "condition": "Cloudy", "humidity": 60},
        "new york":    {"temp_c": 22, "condition": "Clear",  "humidity": 45},
    }
    return json.dumps(data.get(city.lower(), {"error": f"No data for {city}"}))

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather (temperature, condition, humidity) for a city. Available cities: Gandhinagar, Mumbai, Bangalore, Delhi, Chennai, Kolkata, Paris, New York.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Mumbai'"}
            },
            "required": ["city"]
        }
    }
}

print("Weather tool OK:", get_weather("Mumbai"))
"""
        ),
        markdown_cell(
            """### Tool 3: Unit converter

Is 100°F hot or cold? Most people need to convert to answer that. LLMs approximate unit conversions but often get the decimals wrong. A tool gives exact answers.
"""
        ),
        code_cell(
            """# --- Tool 3: Unit converter (precision matters) ---
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    \"\"\"Convert between common units.\"\"\"
    conversions = {
        ("celsius", "fahrenheit"):  lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"):  lambda v: (v - 32) * 5/9,
        ("kg", "pounds"):           lambda v: v * 2.20462,
        ("pounds", "kg"):           lambda v: v / 2.20462,
        ("km", "miles"):            lambda v: v * 0.621371,
        ("miles", "km"):            lambda v: v / 0.621371,
        ("meters", "feet"):         lambda v: v * 3.28084,
        ("feet", "meters"):         lambda v: v / 3.28084,
        ("liters", "gallons"):      lambda v: v * 0.264172,
        ("gallons", "liters"):      lambda v: v / 0.264172,
    }
    key = (from_unit.lower(), to_unit.lower())
    fn = conversions.get(key)
    if fn is None:
        return f"Cannot convert {from_unit} to {to_unit}"
    return f"{fn(value):.2f} {to_unit}"

converter_tool = {
    "type": "function",
    "function": {
        "name": "convert_units",
        "description": "Convert a value between common units. Supports: celsius/fahrenheit, kg/pounds, km/miles, meters/feet, liters/gallons.",
        "parameters": {
            "type": "object",
            "properties": {
                "value":     {"type": "number", "description": "The numeric value to convert"},
                "from_unit": {"type": "string", "description": "Source unit, e.g. 'celsius'"},
                "to_unit":   {"type": "string", "description": "Target unit, e.g. 'fahrenheit'"}
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    }
}

print("Converter tool OK:", convert_units(100, "celsius", "fahrenheit"))
"""
        ),
        markdown_cell(
            """### Tool 4: Course notes search — your own knowledge base

*"What week did we cover data drift?"* — the LLM doesn't know your specific course. This tool is a tiny **RAG system** (Retrieval-Augmented Generation): the tool *retrieves* matching topics from a dictionary, and the model *generates* a natural-language answer from the retrieved data.
"""
        ),
        code_cell(
            """# --- Tool 4: Course notes search (private knowledge base) ---
def search_notes(query: str) -> str:
    \"\"\"Search CS 203 course topics by keyword.\"\"\"
    topics = {
        "data drift":     "Week 10: detecting distribution shift with KS test, PSI, chi-squared test",
        "profiling":      "Week 11: cProfile, timeit, finding bottlenecks in ML code",
        "quantization":   "Week 11: INT8 / FP16, dynamic quantization, ONNX, model compression",
        "pruning":        "Week 11: unstructured and structured pruning, removing weights",
        "distillation":   "Week 11: teacher-student training, soft labels, knowledge transfer",
        "docker":         "Week 10: containerization, Dockerfiles, reproducible environments",
        "fastapi":        "Week 12: building REST APIs, Pydantic validation, /predict endpoints",
        "agents":         "Week 12: tool calling, Gemma 4, the agent loop, function calling",
        "git":            "Week 9: version control, commits, branches, merge conflicts",
        "experiment tracking": "Week 8: MLflow, Weights & Biases, hyperparameter tuning",
        "cross validation":    "Week 7: k-fold CV, train/val/test splits, bias-variance tradeoff",
        "gradio":         "Week 12: building demo UIs for ML models, share=True for public link",
        "streamlit":      "Week 12: building dashboard-style ML apps with Python",
        "batching":       "Week 11: processing multiple inputs at once for throughput",
        "onnx":           "Week 11: portable model format, export once run anywhere",
    }
    query_lower = query.lower()
    matches = {k: v for k, v in topics.items() if query_lower in k or query_lower in v.lower()}
    if matches:
        return json.dumps(matches, indent=2)
    return json.dumps({"message": f"No results for '{query}'. Try: {', '.join(list(topics.keys())[:5])}"})

search_tool = {
    "type": "function",
    "function": {
        "name": "search_notes",
        "description": "Search CS 203 'Software Tools and Techniques for AI' course topics. Returns the week number and key concepts for matching topics.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword or topic to search, e.g. 'docker', 'quantization', 'git'"}
            },
            "required": ["query"]
        }
    }
}

print("Search tool OK:", search_notes("quantization"))
"""
        ),
        markdown_cell(
            """### Testing tool selection

We now have four tools. Let's give the model **all four** at once and see if it picks the right one for each question. This is the critical test — can the model read the "menu" and "order" correctly?
"""
        ),
        code_cell(
            """ALL_TOOLS = [calculator_tool, weather_tool, converter_tool, search_tool]
TOOL_FUNCTIONS = {
    "calculate": calculate,
    "get_weather": get_weather,
    "convert_units": convert_units,
    "search_notes": search_notes,
}

# Test: the model should pick get_weather
messages = [{"role": "user", "content": "Is it raining in Bangalore?"}]
result = generate(messages, tools=ALL_TOOLS)
print("Tool picked:", result["parsed"].get("tool_calls", "none — answered directly"))
"""
        ),
        code_cell(
            """# Test: the model should pick convert_units
messages = [{"role": "user", "content": "I weigh 70 kg. What's that in pounds?"}]
result = generate(messages, tools=ALL_TOOLS)
print("Tool picked:", result["parsed"].get("tool_calls", "none — answered directly"))
"""
        ),
        code_cell(
            """# Test: the model should pick search_notes
messages = [{"role": "user", "content": "What week did we cover Docker?"}]
result = generate(messages, tools=ALL_TOOLS)
print("Tool picked:", result["parsed"].get("tool_calls", "none — answered directly"))
"""
        ),
        code_cell(
            """# Test: the model should NOT use any tool
messages = [{"role": "user", "content": "What is the capital of France?"}]
result = generate(messages, tools=ALL_TOOLS)
if result["parsed"].get("tool_calls"):
    print("Used tool:", result["parsed"]["tool_calls"])
else:
    print("Answered directly:", result["text"])
"""
        ),
        markdown_cell(
            """**If the model picked the right tool every time, congratulations — you have a model that can read a menu and order correctly.**

Notice the last test: for "What is the capital of France?", the model should answer directly *without calling any tool*. It knows the answer from training data, so no tool is needed. **A good agent knows when NOT to use a tool.**

---

## Step 5: The complete agent loop

So far we've been doing tool calls manually — calling `generate()`, extracting the tool call, executing it, feeding the result back, calling `generate()` again. That's tedious.

The **agent loop** automates this entire cycle. Here's the pseudocode:

```
messages = [user's question]

repeat (up to max_steps):
    response = model.generate(messages, tools=...)

    if response has NO tool call:
        return response as final answer     ← done!

    for each tool_call in response:
        result = execute(tool_call)         ← YOUR code runs
        append result to messages

    go back to top of loop                  ← model sees the result
```

That's the entire architecture. Let's implement it:
"""
        ),
        code_cell(
            """def agent(user_message, tools=ALL_TOOLS, tool_functions=TOOL_FUNCTIONS,
         max_steps=5, verbose=True):
    \"\"\"
    A simple agent: ask the model, execute tool calls, repeat until done.

    Args:
        user_message: The user's question (string)
        tools: List of tool schemas the model can choose from
        tool_functions: Dict mapping tool name → Python function
        max_steps: Safety limit to prevent infinite loops
        verbose: If True, print each step for educational purposes
    \"\"\"
    messages = [{"role": "user", "content": user_message}]

    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\\n{'='*60}")
            print(f"Step {step}")
            print(f"{'='*60}")

        # 1. Ask the model (with tools available)
        result = generate(messages, tools=tools)

        # 2. Check: did the model want to call a tool?
        if not result["parsed"].get("tool_calls"):
            # No tool call → this is the final answer
            if verbose:
                print(f"Final answer: {result['text']}")
            return result["text"]

        # 3. Execute each tool call
        messages.append({
            "role": "assistant",
            "tool_calls": result["parsed"]["tool_calls"]
        })

        for tc in result["parsed"]["tool_calls"]:
            name = tc["name"]
            args = tc["arguments"]
            if verbose:
                print(f"  Tool call: {name}({json.dumps(args)})")

            # Execute the function
            if name in tool_functions:
                tool_result = tool_functions[name](**args)
            else:
                tool_result = f"Error: unknown tool '{name}'"

            if verbose:
                print(f"  Result:    {tool_result}")

            messages.append({
                "role": "tool",
                "name": name,
                "content": str(tool_result),
            })

    return "Reached max steps without a final answer."
"""
        ),
        markdown_cell(
            """### Let's try it!

Run each cell below and **watch the step-by-step output**. The `verbose=True` flag shows you exactly what the agent is doing at each step — which tool it calls, what arguments it passes, and what result it gets back.
"""
        ),
        code_cell(
            """# Simple: one tool call — should use calculate
agent("What is 4729 times 8314?")
"""
        ),
        code_cell(
            """# Weather: should use get_weather (possibly twice to compare)
agent("What's the temperature in Delhi right now? Is it hotter than Mumbai?")
"""
        ),
        code_cell(
            """# Unit conversion: should use convert_units
agent("Convert 5 miles to kilometers.")
"""
        ),
        code_cell(
            """# Course notes: should use search_notes
agent("Which week covered experiment tracking?")
"""
        ),
        code_cell(
            """# No tool needed — should answer from memory
agent("What does HTML stand for?")
"""
        ),
        markdown_cell(
            """### Multi-step reasoning: questions that need multiple tool calls

These are the fun ones — the model has to **plan**, call **multiple tools** in sequence, and **combine** the results into a single answer.

Watch how the agent uses 2-4 steps to answer questions that require chaining different tools together:
"""
        ),
        code_cell(
            """# Needs: get_weather + convert_units
agent("What's the temperature in Gandhinagar in Fahrenheit?")
"""
        ),
        code_cell(
            """# Needs: get_weather (x2) + calculate
agent("How much hotter is Delhi than Bangalore right now, in degrees Celsius?")
"""
        ),
        code_cell(
            """# Needs: calculate + convert_units
agent("If I run 5 km every day for a week, how many miles is that total?")
"""
        ),
        markdown_cell(
            """---

## Step 6: Add your own tool

Now it's your turn! **Writing a new tool** is the core skill of an AI engineer building agents. The process is always the same:

1. **Write a Python function** that does the thing you want
2. **Write a JSON schema** describing the function (name, description, parameters)
3. **Add both** to the agent's toolkit
4. **Test** that the model uses it correctly

Below is a starter example (a CS/ML dictionary). You can modify it or write something completely different — a calorie counter, a timezone converter, a joke generator, a recipe lookup, etc.
"""
        ),
        code_cell(
            """# TODO — define your own tool function and schema.
# Below is a starter example. You can modify it or write something
# completely different (e.g. a calorie counter, a timezone converter,
# a joke generator, a recipe lookup, etc.)

def define_word(word: str) -> str:
    \"\"\"Look up the definition of a common CS / ML term.\"\"\"
    definitions = {
        "overfitting":  "When a model learns the training data too well, including noise, and performs poorly on new data.",
        "gradient":     "The vector of partial derivatives of the loss function with respect to each parameter.",
        "epoch":        "One full pass through the entire training dataset.",
        "batch size":   "The number of training examples used in one forward/backward pass.",
        "learning rate": "A hyperparameter that controls how much to adjust the model's weights during training.",
        "regularization": "Techniques to prevent overfitting, e.g. L1/L2 penalties, dropout.",
        "transformer":  "A neural network architecture based on self-attention, used in GPT, BERT, etc.",
        "tokenizer":    "Converts text into a sequence of integer IDs that a model can process.",
        # TODO: add more terms here!
    }
    result = definitions.get(word.lower())
    if result:
        return json.dumps({"word": word, "definition": result})
    return json.dumps({"error": f"'{word}' not found. Available: {', '.join(definitions.keys())}"})


# TODO — fill in the schema for your tool
dictionary_tool = {
    "type": "function",
    "function": {
        "name": "define_word",
        "description": "Look up the definition of a common computer science or machine learning term.",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The CS/ML term to define, e.g. 'overfitting', 'gradient'"
                }
            },
            "required": ["word"]
        }
    }
}

# Test it directly
print(define_word("overfitting"))
"""
        ),
        code_cell(
            """# Add your new tool to the agent's toolkit
MY_TOOLS = ALL_TOOLS + [dictionary_tool]
MY_FUNCTIONS = {**TOOL_FUNCTIONS, "define_word": define_word}

# Test: does the agent use your tool?
agent("What does overfitting mean?", tools=MY_TOOLS, tool_functions=MY_FUNCTIONS)
"""
        ),
        code_cell(
            """# Test: a question that combines your tool with another
agent("What does 'epoch' mean, and if I train for 10 epochs with batch size 32 over 1000 samples, how many forward passes is that?",
      tools=MY_TOOLS, tool_functions=MY_FUNCTIONS)
"""
        ),
        markdown_cell(
            """---

## Step 7: The big picture

### What you built

You just built an AI agent from scratch. Let's recap the three pieces:

| Piece | What it does | Analogy |
|:--|:--|:--|
| **LLM** (Gemma 4) | Thinks, reasons, decides what to do next | The brain |
| **Tools** (4 functions) | Functions the LLM can call | The hands |
| **Loop** (`agent()`) | Keeps going until the task is done | The work ethic |

**Total: about 100 lines of Python.** That's the entire agent.

### This is the architecture behind every AI agent

| Agent | What it does | Tools it uses |
|:--|:--|:--|
| **Claude Code** | Writes, edits, and tests code from your terminal | File read/write, bash, grep, git |
| **Cursor / Windsurf** | AI-powered code editors | File system, LSP, terminal |
| **Devin** | Autonomous software engineer | Browser, terminal, code editor |
| **Perplexity** | Search engine with citations | Web search, web scrape |
| **Google Deep Research** | Multi-step research reports | Search, summarize, cite |

Every one of these is the **same pattern**: LLM + tools + loop. The tools are more powerful, the models are bigger, but the architecture is identical to what you just built.

### Guardrails: when agents go wrong

Our `max_steps=5` parameter is the simplest guardrail. Real-world agents also need:

| Risk | Example | Guardrail |
|:--|:--|:--|
| **Infinite loop** | Agent keeps calling tools forever | `max_steps` parameter |
| **Wrong tool** | Agent calls `delete_file` when it meant `read_file` | Require user confirmation for dangerous tools |
| **Hallucinated args** | `get_weather("Narnia")` | Validate arguments before execution |
| **Cost explosion** | Agent makes 1000 API calls | Budget / rate limiting |

### Reflection questions

Answer these in the markdown cells below.
"""
        ),
        markdown_cell(
            """**Q1:** When you asked "What is 4729 times 8314?", the model called the
calculator tool instead of answering directly. Why? What would have happened
without the tool?

> _Your answer here_
"""
        ),
        markdown_cell(
            """**Q2:** When you asked "What is the capital of France?", the model answered
directly without calling any tool. How did it decide not to use a tool?

> _Your answer here_
"""
        ),
        markdown_cell(
            """**Q3:** The `max_steps=5` parameter is a guardrail. What would happen if
you removed it (set it to infinity) and the model kept calling tools in a
loop? Give a concrete scenario where this could go wrong.

> _Your answer here_
"""
        ),
        markdown_cell(
            """**Q4:** Our `get_weather` function returns mock data. If you wanted to make
it return *real* weather data, what would you change? (1-2 sentences, no code needed)

> _Your answer here_
"""
        ),
        markdown_cell(
            """**Q5:** Name one real-world application (outside this course) where an
agent with tool use would be more useful than a plain LLM. What tools
would it need?

> _Your answer here_
"""
        ),
        markdown_cell(
            """---

## Bonus: Try these challenges

If you have time, try these (no auto-grading — just for fun and learning):

1. **Add a `get_time` tool** that returns the current time using
   `datetime.datetime.now()`. Ask the agent "What time is it?" and
   "How many hours until midnight?"

2. **Add a `translate` tool** with a small hard-coded dictionary of
   English → Hindi translations. Ask "How do you say 'hello' in Hindi?"

3. **Break the agent.** Ask questions that are adversarial or confusing.
   Can you make the model call the wrong tool? What happens when you ask
   a question about a city that's not in the weather database?

4. **Chain 3+ tools.** Write a question that forces the model to call at
   least three different tools to answer. Does it succeed?

---

## Going further: MCP — the universal tool standard

In this notebook, you defined tools as Python dicts. But what if you want to share tools across different LLMs and applications?

**MCP (Model Context Protocol)** is an open standard from Anthropic that lets anyone publish tools that any agent can use — like USB-C for AI tools. One tool definition, many LLMs.

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Claude  │     │ Gemma 4  │     │  GPT-5   │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
     └────────┬───────┘────────┬───────┘
              │                │
         ┌────┴────┐     ┌────┴────┐
         │ Weather │     │  Slack  │
         │  MCP    │     │  MCP    │
         │ Server  │     │ Server  │
         └─────────┘     └─────────┘
```

This is an active and exciting area — the tools you build today could be published as MCP servers that anyone in the world can use with any LLM.
"""
        ),
    ]


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_notebook("01-agents-from-scratch.ipynb", nb_agents())


if __name__ == "__main__":
    main()

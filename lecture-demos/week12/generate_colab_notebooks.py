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

This cell takes ~1 minute. It installs the latest `transformers` and
`bitsandbytes` for 4-bit quantization.
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

Gemma 4 E2B has 2 billion parameters. At full precision (FP16) it needs
~4 GB of VRAM. With **4-bit quantization** we cut that to ~1.5 GB, leaving
plenty of room for the T4's 15 GB.

This cell takes ~60 seconds the first time (downloading the model) and
~15 seconds on subsequent runs (loading from cache).
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

Before we add tools, let's confirm the model works with a simple question.
This is a plain LLM call — the model answers from its training data.
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
        code_cell(
            """messages = [{"role": "user", "content": "What is the capital of India?"}]
result = generate(messages)
print(result["text"])
"""
        ),
        markdown_cell(
            """The model answered from memory. Now let's see what happens when we ask
something it **can't** answer from memory:
"""
        ),
        code_cell(
            """messages = [{"role": "user", "content": "What is the weather in Gandhinagar right now?"}]
result = generate(messages)
print(result["text"])
"""
        ),
        markdown_cell(
            """The model probably said something like *"I don't have access to real-time
data"* or it guessed based on general knowledge.

**This is the problem we're going to solve.** We'll give the model a
`get_weather` tool so it can look up real data instead of guessing.

---

## Step 3: Your first tool — the calculator

### Why does an LLM need a calculator?

LLMs predict tokens, they don't compute. Ask a hard math question and
they'll confidently give the wrong answer. Let's see:
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
            """Probably wrong! Let's fix this by giving the model a calculator tool.

### Defining the tool

A tool has two parts:
1. A **Python function** that does the actual work
2. A **JSON schema** that tells the model what the function does and what arguments it expects
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

Now we pass `tools=[calculator_tool]` when generating. The model will see
the tool description and **decide** whether to use it.
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
            """You should see a `tool_calls` list with `"name": "calculate"` and
`"arguments": {"expression": "4729 * 8314"}`.

The model did NOT compute the answer — it asked US to compute it. Let's
execute the tool and feed the result back:
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

**Recap of what just happened:**
1. We asked a math question
2. The model decided to use the calculator tool (instead of guessing)
3. It formatted the arguments correctly as a JSON object
4. We executed the function and got the result
5. We fed the result back to the model
6. The model wrote a natural-language answer using the real result

---

## Step 4: Three more tools

Let's build a toolkit of four tools. Together, these will make a surprisingly
capable agent.
"""
        ),
        code_cell(
            """# --- Tool 2: Weather lookup ---
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
        code_cell(
            """# --- Tool 3: Unit converter ---
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
        code_cell(
            """# --- Tool 4: Course notes search ---
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
            """### Let's test each tool individually

Before building the full agent loop, let's verify the model picks the right
tool for different questions.
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
            """**If the model picked the right tool every time, congratulations — you have
a model that can read a menu and order correctly.**

Now let's build the loop that actually *executes* the tools and feeds the
results back.

---

## Step 5: The complete agent loop

This is the entire agent, in one function. Read every line — there's no magic.
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
"""
        ),
        code_cell(
            """# Simple: one tool call
agent("What is 4729 times 8314?")
"""
        ),
        code_cell(
            """# Weather question
agent("What's the temperature in Delhi right now? Is it hotter than Mumbai?")
"""
        ),
        code_cell(
            """# Unit conversion
agent("Convert 5 miles to kilometers.")
"""
        ),
        code_cell(
            """# Course notes
agent("Which week covered experiment tracking?")
"""
        ),
        code_cell(
            """# No tool needed — should answer from memory
agent("What does HTML stand for?")
"""
        ),
        markdown_cell(
            """### Multi-step: questions that need multiple tool calls

These are the fun ones — the model has to plan, call multiple tools, and
combine the results.
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

Now it's your turn. Write a new tool, add it to the agent, and test it.

### Example: a dictionary / definition lookup tool
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

## Step 7: Reflection and submission

### What you built

You just built an AI agent from scratch. Let's recap the pieces:

| Component | What it does | Lines of code |
|:--|:--|:--|
| `generate()` | Wraps the model call + parsing | ~15 lines |
| Tool functions | `calculate`, `get_weather`, `convert_units`, `search_notes` | ~10 lines each |
| Tool schemas | JSON descriptions of each tool | ~15 lines each |
| `agent()` | The loop: generate → check for tool call → execute → repeat | ~30 lines |

**Total: about 100 lines of Python.** That's the entire agent.

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

If you have time, try these (no auto-grading — just for fun):

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

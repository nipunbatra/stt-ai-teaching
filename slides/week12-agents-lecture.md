---
marp: true
theme: iitgn-modern
paginate: true
---

<!-- _class: title-slide -->
<!-- _paginate: false -->

# Building AI Agents from Scratch

## Week 12: CS 203 - Software Tools and Techniques for AI

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# Where We Are in the Course

```
Week 7-8:   Build and tune the model      → Is it good?
Week 9:     Reproducibility                → Can someone re-run it?
Week 10:    Data drift                     → Is it STILL good?
Week 11:    Profiling & Quantization       → Is it FAST and SMALL?
Week 12:    Agents from Scratch            → Can it DO things?   ← today
```

Weeks 7-11 were about building and shipping a model that *answers questions*.

Today we build something that **takes actions**.

---

# A Simple Question, A Hard Answer

You ask an LLM: *"What's the weather in Gandhinagar right now?"*

The LLM replies:

> "I don't have access to real-time data, but Gandhinagar typically
> experiences temperatures around 35-40°C in April..."

It *guessed*. It didn't *check*. Why?

**An LLM is just a text predictor.** It can only generate the next word.
It can't open a browser, call an API, run code, or check a database.

---

# What If It *Could* Call an API?

Imagine the LLM could say:

> "I need to check the weather. Let me call the weather API."

Then it produces this:

```json
{"tool": "get_weather", "arguments": {"city": "Gandhinagar"}}
```

Your code executes that, gets back `{"temp": 38, "condition": "Sunny"}`,
and feeds it to the LLM, which then says:

> "It's currently 38°C and sunny in Gandhinagar."

**That's an agent.** An LLM that can *use tools*.

---

# What Is an Agent?

An **agent** = LLM + tools + a loop.

```
┌──────────────────────────────────────────────────┐
│                   The Agent Loop                 │
│                                                  │
│   User question                                  │
│        │                                         │
│        ▼                                         │
│   ┌─────────┐     "I need to call a tool"        │
│   │   LLM   │ ─────────────────────────┐         │
│   │ (think) │                          │         │
│   └────┬────┘                          ▼         │
│        │                        ┌──────────┐     │
│        │ "I have the answer"    │ Tool     │     │
│        │                        │ (act)    │     │
│        ▼                        └────┬─────┘     │
│   Final answer                      │            │
│   to the user              result goes back      │
│                              to the LLM          │
└──────────────────────────────────────────────────┘
```

The LLM **decides** whether to use a tool. You **execute** the tool. Repeat.

---

# You Already Use Agents Every Day

| App | What you say | What the agent *does* |
|:--|:--|:--|
| Google Assistant | "Set a timer for 10 minutes" | Calls the clock API |
| Siri | "Text Mom I'll be late" | Calls the messaging API |
| ChatGPT with browsing | "Summarize today's news" | Calls the Bing search API |
| Claude Code | "Fix the failing test" | Reads files, edits code, runs pytest |
| GitHub Copilot | "Add error handling here" | Reads context, writes code inline |

In every case: an LLM **decides** what action to take, your device
**executes** it, and the result goes **back to the LLM** for the next step.

---

# Three Things That Make an Agent

| Piece | What it does | Analogy |
|:--|:--|:--|
| **LLM** | Thinks, reasons, decides what to do next | The brain |
| **Tools** | Functions the LLM can call (calculator, search, code runner) | The hands |
| **Loop** | Keeps going until the task is done | The work ethic |

Without tools, the LLM can only *talk*. With tools, it can *do*.

Without a loop, it can do *one thing*. With a loop, it can do
*multi-step tasks* — like "find the cheapest flight, then book it."

---

# The "Restaurant" Analogy

**Without tools (plain LLM):**
> You ask the chef: "What's a good recipe for biryani?"
> The chef tells you the recipe from memory. Helpful, but you still need
> to cook it yourself.

**With tools (agent):**
> You ask the chef: "Make me biryani."
> The chef *checks the fridge* (tool: inventory lookup), *orders missing
> spices* (tool: delivery API), *sets the oven* (tool: oven control),
> and *serves you the biryani*.

The chef (LLM) still has the knowledge. The tools let it **act on that knowledge**.

---

<!-- _class: section-divider -->

# Part 1: Running Gemma 4 on Free Colab

*A powerful open model that fits on a free GPU.*

---

# Why Gemma 4?

| Feature | What it means for us |
|:--|:--|
| **Open weights** | Download and run locally — no API key needed |
| **Apache 2.0 license** | Free for any use, including commercial |
| **E2B model (2B params)** | Fits on a free Colab T4 GPU at 4-bit |
| **Native tool calling** | Built-in support — no prompt hacking needed |
| **128K context window** | Can read and reason over long documents |

Other options exist (Llama, Mistral, Qwen), but Gemma 4 has the cleanest
tool-calling support and runs on the smallest hardware.

---

# Loading Gemma 4 E2B on Colab — The Setup

```python
!pip install -U transformers accelerate bitsandbytes -q

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-e2b-it"

# 4-bit quantization → fits in ~3 GB instead of ~8 GB
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
```

This takes about 60 seconds on a T4. After that, generation is fast.

---

# Your First Generation

```python
messages = [
    {"role": "user", "content": "What is 2 + 2?"}
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0][inputs.input_ids.shape[-1]:],
                            skip_special_tokens=True)
print(response)
# → "2 + 2 = 4"
```

This is a normal LLM call — no tools yet. The model answered from its
training data. Let's make it smarter.

---

<!-- _class: section-divider -->

# Part 2: Function Calling — Teaching the Model to Use Tools

*The model describes what it wants. You execute it.*

---

# The Key Insight

**Function calling is NOT the model running code.** The model never executes
anything. Here's what actually happens:

```
1. You tell the model: "Here are tools you can use"
     (a list of function names + descriptions)

2. The model reads the user's question and DECIDES:
     "I should call get_weather with city='Gandhinagar'"

3. The model outputs a STRUCTURED request:
     {"name": "get_weather", "arguments": {"city": "Gandhinagar"}}

4. YOUR CODE executes the function and gets the result:
     {"temp": 38, "condition": "Sunny"}

5. You feed the result BACK to the model

6. The model writes the final answer using the real data:
     "It's 38°C and sunny in Gandhinagar."
```

The model is a **decision maker**. You are the **executor**.

---

# Defining a Tool — It's Just a Dictionary

```python
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Mumbai'"
                }
            },
            "required": ["city"]
        }
    }
}
```

This is a **menu** you're handing the model. It tells the model:
- *What* the tool does (`description`)
- *What inputs* the tool needs (`parameters`)
- *What's required* vs optional

---

# How Gemma 4 Calls a Tool

```python
messages = [{"role": "user", "content": "What's the weather in Gandhinagar?"}]

inputs = processor.apply_chat_template(
    messages,
    tools=[weather_tool],          # ← hand the model the menu
    tokenize=True, return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=512)
raw = processor.decode(output[0][inputs.input_ids.shape[-1]:],
                       skip_special_tokens=False)
parsed = processor.parse_response(raw)
```

`parsed` now contains:
```python
{
    "tool_calls": [{"name": "get_weather",
                    "arguments": {"city": "Gandhinagar"}}],
    "content": None
}
```

The model **chose** to call `get_weather`. It did NOT make up the answer.

---

# Feeding the Result Back

```python
import json

# 1. Record the model's decision in the conversation
messages.append({
    "role": "assistant",
    "tool_calls": parsed["tool_calls"]
})

# 2. Execute the tool (YOUR code, not the model's)
result = {"temp_c": 38, "condition": "Sunny"}

# 3. Feed the result back
messages.append({
    "role": "tool",
    "name": "get_weather",
    "content": json.dumps(result)
})

# 4. Let the model write the final answer
inputs = processor.apply_chat_template(
    messages, tools=[weather_tool],
    tokenize=True, return_dict=True,
    return_tensors="pt", add_generation_prompt=True,
).to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
# → "It's currently 38°C and sunny in Gandhinagar."
```

---

<!-- _class: section-divider -->

# Part 3: Four Use Cases

*Each one shows a different kind of tool.*

---

# Use Case 1: Calculator

**Problem:** LLMs are bad at arithmetic. Ask "What is 4729 × 8314?" and the
model will confidently give the wrong answer.

**Fix:** Give it a calculator tool.

```python
def calculate(expression: str) -> str:
    """Evaluate a math expression safely.
    Args:
        expression: A Python math expression like '4729 * 8314'
    """
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"
    return str(eval(expression))
```

Now ask: *"What is 4729 times 8314?"*

The model produces:
```json
{"name": "calculate", "arguments": {"expression": "4729 * 8314"}}
```

Your code runs `eval("4729 * 8314")` → `39,317,006`. The model reports the
exact answer instead of guessing.

---

# Use Case 1: Calculator — Why It Matters

This is not a toy example. Real agents use calculators constantly:

| Scenario | What the agent calculates |
|:--|:--|
| "What's 18% tip on ₹2,450?" | `2450 * 0.18` |
| "How many days until Aug 15?" | Date arithmetic |
| "If I invest ₹10,000 at 8% for 5 years…" | `10000 * (1.08 ** 5)` |
| "Average of these test scores: 82, 91, 76, 88" | `(82+91+76+88) / 4` |

Every time an LLM needs exact arithmetic, it should **call a tool** instead
of trying to compute in its "head" (token prediction is not math).

---

# Use Case 2: Weather Lookup

**Problem:** LLMs have no access to real-time data.

```python
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    Args:
        city: City name, e.g. 'Gandhinagar'
    """
    # In a real app, this calls an API like OpenWeatherMap.
    # For our demo, we use a mock.
    data = {
        "Gandhinagar": {"temp_c": 38, "condition": "Sunny", "humidity": 25},
        "Mumbai":      {"temp_c": 32, "condition": "Humid", "humidity": 80},
        "Bangalore":   {"temp_c": 28, "condition": "Rainy", "humidity": 65},
    }
    return json.dumps(data.get(city, {"error": f"No data for {city}"}))
```

Ask: *"Should I carry an umbrella in Bangalore today?"*

The model calls `get_weather("Bangalore")`, sees `"Rainy"`, and answers:
*"Yes — it's currently rainy in Bangalore with 65% humidity."*

It used **real data** (well, mock data) instead of guessing.

---

# Use Case 3: Unit Converter

**Problem:** Conversion mistakes are common. Is 100°F hot or cold?

```python
import math

def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units.
    Args:
        value: The numeric value to convert
        from_unit: Source unit (e.g. 'celsius', 'kg', 'miles')
        to_unit: Target unit (e.g. 'fahrenheit', 'pounds', 'km')
    """
    conversions = {
        ("celsius", "fahrenheit"):  lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"):  lambda v: (v - 32) * 5/9,
        ("kg", "pounds"):           lambda v: v * 2.20462,
        ("pounds", "kg"):           lambda v: v / 2.20462,
        ("km", "miles"):            lambda v: v * 0.621371,
        ("miles", "km"):            lambda v: v / 0.621371,
    }
    fn = conversions.get((from_unit.lower(), to_unit.lower()))
    if fn is None:
        return f"Cannot convert {from_unit} to {to_unit}"
    return f"{fn(value):.2f} {to_unit}"
```

Ask: *"I weigh 70 kg — what's that in pounds?"*
Model calls `convert_units(70, "kg", "pounds")` → `"154.32 pounds"`.

---

# Use Case 4: Course Notes Search

**Problem:** The student asks *"What week did we cover data drift?"* and the
LLM doesn't know your specific course.

```python
def search_notes(query: str) -> str:
    """Search the CS 203 course topics.
    Args:
        query: A keyword or topic to search for
    """
    topics = {
        "data drift": "Week 10: detecting distribution shift with KS test, PSI",
        "profiling":  "Week 11: cProfile, timeit, finding bottlenecks",
        "quantization": "Week 11: INT8, ONNX, dynamic quantization",
        "docker":     "Week 10: containerization, Dockerfiles, reproducibility",
        "fastapi":    "Week 12: building APIs, Pydantic validation, /predict",
        "agents":     "Week 12: tool calling, Gemma 4, the agent loop",
        "git":        "Week 9: version control, branching, merge conflicts",
    }
    query_lower = query.lower()
    matches = {k: v for k, v in topics.items() if query_lower in k}
    if matches:
        return json.dumps(matches)
    return f"No results for '{query}'"
```

This is a **tiny RAG system**: retrieval (search the dict) + generation
(the LLM writes a natural-language answer using the retrieved info).

---

# All Four Tools Together

Give the model **all four tools at once**, and it picks the right one:

| User question | Tool the model picks |
|:--|:--|
| "What's 17 squared?" | `calculate("17 ** 2")` |
| "Is it raining in Mumbai?" | `get_weather("Mumbai")` |
| "Convert 100°F to Celsius" | `convert_units(100, "fahrenheit", "celsius")` |
| "When did we learn about Docker?" | `search_notes("docker")` |
| "What is the capital of France?" | *No tool* — answers from memory |

The model reads the descriptions you gave it and **chooses which tool fits
the question**, or answers directly if no tool is needed.

---

<!-- _class: section-divider -->

# Part 4: The Agent Loop

*Ten lines of code that tie everything together.*

---

# The Complete Agent Loop

```python
def agent(user_message, tools, tool_functions, max_steps=5):
    messages = [{"role": "user", "content": user_message}]

    for step in range(max_steps):
        # 1. Ask the model
        inputs = processor.apply_chat_template(
            messages, tools=tools, tokenize=True,
            return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=512)
        raw = processor.decode(output[0][inputs.input_ids.shape[-1]:],
                               skip_special_tokens=False)
        parsed = processor.parse_response(raw)

        # 2. If no tool call → we have the final answer
        if not parsed.get("tool_calls"):
            return processor.decode(output[0][inputs.input_ids.shape[-1]:],
                                    skip_special_tokens=True)

        # 3. Execute each tool call
        messages.append({"role": "assistant",
                         "tool_calls": parsed["tool_calls"]})
        for tc in parsed["tool_calls"]:
            result = tool_functions[tc["name"]](**tc["arguments"])
            messages.append({"role": "tool", "name": tc["name"],
                             "content": str(result)})

    return "Reached max steps without a final answer."
```

---

# Walking Through a Multi-Step Example

Ask: *"What's the square root of 144 plus the temperature in Mumbai in Fahrenheit?"*

```
Step 1:  Model thinks → calls calculate("144 ** 0.5")
         Result: "12.0"

Step 2:  Model thinks → calls get_weather("Mumbai")
         Result: {"temp_c": 32, "condition": "Humid"}

Step 3:  Model thinks → calls convert_units(32, "celsius", "fahrenheit")
         Result: "89.60 fahrenheit"

Step 4:  Model thinks → calls calculate("12.0 + 89.60")
         Result: "101.6"

Step 5:  Model answers: "The square root of 144 is 12, and the temperature
         in Mumbai is 32°C (89.6°F). Added together: 101.6."
```

**Four tool calls, three different tools, one coherent answer.** The model
*planned* which tools to call and in what order — you just executed them.

---

# What Just Happened?

The model did five things that a plain LLM can't:

1. **Decomposed** a compound question into sub-problems
2. **Selected** the right tool for each sub-problem
3. **Formatted** the arguments correctly (JSON)
4. **Waited** for each result before proceeding
5. **Synthesized** all the results into one answer

The model didn't "learn" how to do math or check weather.
**You gave it hands (tools) and it figured out how to use them.**

---

<!-- _class: section-divider -->

# Part 5: Where This Is Going

*From toy demos to real-world impact.*

---

# Agents in the Wild (2025-2026)

| Agent | What it does | Tools it uses |
|:--|:--|:--|
| **Claude Code** | Writes, edits, and tests code from your terminal | File read/write, bash, grep, git |
| **Cursor / Windsurf** | AI-powered code editors | File system, LSP, terminal |
| **Devin** | Autonomous software engineer | Browser, terminal, code editor |
| **Perplexity** | Search engine with citations | Web search, web scrape |
| **Google Deep Research** | Multi-step research reports | Search, summarize, cite |
| **OpenAI Operator** | Uses websites on your behalf | Browser automation |

Every one of these is the **same pattern** you just learned:
LLM + tools + loop. The tools are more powerful, the models are bigger,
but the architecture is identical.

---

# MCP — The Universal Tool Standard

Today you defined tools as Python dicts. But what if you want to share tools
across different LLMs and applications?

**MCP (Model Context Protocol)** is an open standard from Anthropic that
lets anyone publish tools that any agent can use:

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

One tool definition, many LLMs. Like USB-C for AI tools.

---

# The Agent Stack — A Mental Model

```
Layer 4:  Application      Claude Code, Cursor, Devin, your app
                           ↑
Layer 3:  Agent framework  The loop + memory + planning
                           ↑
Layer 2:  Tools            Calculator, search, file I/O, APIs
                           ↑
Layer 1:  LLM              Gemma 4, Claude, GPT, Llama
                           ↑
Layer 0:  Hardware          Your laptop, Colab T4, cloud GPU
```

**Today you built layers 1-3 from scratch.** Layer 4 is what you build
when you put it all together into a product.

The entire field of "AI engineering" in 2025-2026 is about making this
stack reliable, fast, and safe.

---

# Guardrails: When Agents Go Wrong

Agents can be powerful but also unpredictable. Real-world agents need
**guardrails**:

| Risk | Example | Guardrail |
|:--|:--|:--|
| **Infinite loop** | Agent keeps calling tools forever | `max_steps` parameter |
| **Wrong tool** | Agent calls `delete_file` when it meant `read_file` | Require user confirmation for dangerous tools |
| **Hallucinated args** | `get_weather("Narnia")` | Validate arguments before execution |
| **Cost explosion** | Agent makes 1000 API calls | Budget / rate limiting |

Our `max_steps=5` parameter is the simplest guardrail. Production agents
add human-in-the-loop confirmation, sandboxing, and cost limits.

---

<!-- _class: section-divider -->

# Summary

---

# Key Takeaways

1. **An agent = LLM + tools + loop.** The LLM *decides*, you *execute*.

2. **Function calling** is how the model requests tool use — it produces
   structured JSON, not free-form text.

3. **Tools are just Python functions** with a description. The model
   reads the description and decides when to use each one.

4. **The agent loop** is ~10 lines: generate → check for tool call →
   execute → feed result back → repeat.

5. **Gemma 4 E2B** runs on a free Colab T4 with 4-bit quantization and
   has native tool-calling support.

6. **This is the architecture behind Claude Code, Cursor, Devin,
   Perplexity**, and every other AI agent you use. Same pattern, bigger tools.

---

# What to Try in the Notebook

| Task | What you'll do |
|:--|:--|
| Load Gemma 4 | Run a 4-bit model on free Colab T4 |
| Single tool call | Define a calculator tool, watch the model use it |
| Multi-tool agent | Give the model 4 tools, ask compound questions |
| Build your own tool | Write a new function, add it to the agent |
| Multi-step reasoning | Ask questions that need 3+ tool calls |

> [Open the notebook in Colab](https://colab.research.google.com/github/nipunbatra/stt-ai-teaching/blob/master/lecture-demos/week12/colab-notebooks/01-agents-from-scratch.ipynb)

---

<!-- _class: title-slide -->

# Questions?

## LLM + Tools + Loop = Agent.


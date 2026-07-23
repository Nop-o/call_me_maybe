*This project has been created as part of the 42 curriculum by [adamez-f](https://github.com/Nop-o).*

# call_me_maybe

## Table of Contents
- [call\_me\_maybe](#call_me_maybe)
	- [Table of Contents](#table-of-contents)
	- [Description](#description)
	- [The Generation Pipeline](#the-generation-pipeline)
	- [Algorithm Explanation](#algorithm-explanation)
	- [Design Decisions](#design-decisions)
	- [Performance Analysis](#performance-analysis)
	- [Challenges Faced](#challenges-faced)
	- [Testing Strategy](#testing-strategy)
	- [Instructions](#instructions)
		- [All available commands](#all-available-commands)
	- [Example Usage](#example-usage)
	- [Resources](#resources)
		- [AI Usage](#ai-usage)

---

## Description

`call_me_maybe` is a Python program that uses a large language model (LLM) to understand natural language prompts and automatically call the right function with the right arguments. Given a list of prompts and a set of function definitions in JSON, it figures out which function best matches each prompt and extracts the parameters needed to run it. The main challenge is getting the model to produce consistent, structured outputs from unpredictable human input.

The program works with two JSON files. The first contains a list of prompts, each following a three-tag structure with a `system`, `user`, and `assistant` field. The second contains a list of function definitions, each describing a function's name, its parameters and their types, and what it returns. The program reads both files internally, and for each prompt, uses an LLM to determine which function best matches the intent and what arguments to pass to it.

---

## The Generation Pipeline

The LLM generation process follows these steps:

1. **Prompt** — A tagged prompt is fed to the model as structured input.

2. **Tokenization** — The text is broken into subword units called tokens. Unlike simple word splitting, tokenizers handle punctuation, leading spaces, and split words into smaller components using algorithms like BPE or SentencePiece.

3. **Input IDs** — Each token is converted into a numerical ID the model can process internally.

4. **LLM Processing** — The model processes these IDs through its neural network layers.

5. **Logits** — The model outputs a probability score for every token in its vocabulary, representing how likely each one is to come next.

6. **Token Selection** — The token with the highest probability is selected. This is where **constrained decoding** is applied, restricting valid choices to ensure the output always follows a specific structure — such as producing a valid function call.

This process repeats token by token. Each newly generated token is appended to the input, and steps 2–6 repeat until the full response is produced.

```
Prompt → Tokenization → Input IDs → LLM → Logits → Token Selection → [repeat]
```

---

## Algorithm Explanation

To ensure a high percentage of correct answers from the LLM, constrained decoding is used. Instead of letting the model generate any token freely, it restricts which tokens are accessible at each generation step.

The model used (Qwen3-0.6B) has 500 million parameters. With constrained decoding, only a selective portion of its vocabulary is made available depending on what is being generated:

- **Function name** — The model can only choose tokens that are part of the known function names loaded from the JSON file.
- **Parameters** — Each parameter type restricts the available tokens accordingly. For example, an integer parameter only allows digit tokens and the `-` sign, while a string parameter opens a broader but still controlled token set.

This approach guarantees that every output is a structurally valid function call, regardless of how the model would naturally respond.

---

## Design Decisions

The core design choice was to use constrained decoding to force the model to output only valid function names, rather than parsing free-form text after generation. Functions are loaded at runtime from the JSON file, making the program agnostic to any specific set of functions.

Each prompt in the prompts file follows a three-tag structure:
- A `system` tag that sets the model's behavior and context.
- A `user` tag containing the actual natural language instruction.
- An `assistant` tag representing the expected model output — the ground truth function call.

This structure mirrors the standard chat completion format and allows the program to feed conversation context directly to the LLM in a consistent and predictable way.

---

## Performance Analysis

The model performs well on simple, unambiguous prompts with numeric parameters. Accuracy drops on prompts involving string manipulation or complex reasoning, where the model's output can become unpredictable and produce hallucinations. Speed depends on the model size and the number of prompts to process.

---

## Challenges Faced

The main challenge was ensuring the model always produces a valid function call rather than a generic text response. Getting the argument types to match the expected function signature — especially for string and regex parameters — required careful tuning of the constrained token sets and thorough testing of edge cases.

---

## Testing Strategy

The implementation was validated by running a set of hand-crafted prompts with known expected outputs and comparing the model's function calls against them. Edge cases such as ambiguous or empty prompts were tested separately to verify that the program handles failures gracefully.

---

## Instructions

To get started, use the provided `Makefile`:

```bash
make install
make run
```

### All available commands

| Command | Description |
| :--- | :--- |
| `make install` | Create virtual environment and install dependencies |
| `make clean` | Remove all temporary files and caches |
| `make run` | Execute the main script |
| `make lint` | Run flake8 and mypy with standard checks |
| `make lint-strict` | Run flake8 and mypy with strict mode |
| `make debug` | Run the main script in debug mode (pdb) |
| `make help` | Show a help message |

---

## Example Usage

Run the following commands to install dependencies and execute the program:

```bash
make install
make run
```

For a prompt like `"What is the sum of 2 and 3?"`, the program will identify the matching function and output the corresponding call with the correct arguments.

---

## Resources

[Blog: Model File Format](https://blog.devops.dev/understanding-hugging-face-model-file-formats-ggml-and-gguf-914b0ebd1131)

[w3schools: Python built-in functions reference](https://www.w3schools.com/python/)

[3Blue1Brown: YouTube video about LLMs](https://www.youtube.com/watch?v=LPZh9BOjkQs)

[Json informations](https://www.json.org/json-fr.html)

### AI Usage

AI was used throughout this project for:
- **Code review**: spotting unused variables, dead code, and inconsistent type annotations.
- **README**: writing and structuring this document based on project content.
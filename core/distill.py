"""
distill.py

Ollama-based reasoning agent for graph distillation.

A dedicated agent whose sole job is to grow the graph.
It reasons through mathematical topics step by step,
and every reasoning chain gets ingested into the graph.

The agent:
    1. Picks a topic from the curriculum
    2. Asks the model to reason through it step by step
    3. Extracts the reasoning chain (especially <think> blocks)
    4. Scores each completion with the reward function
    5. Ingests high-reward chains as positive nodes/edges
    6. Ingests low-reward chains as contrastive edges
    7. Queries the graph to find weak areas and focuses there next

The model is the teacher. The graph is the student.
The collision loop runs alongside, finding connections
the agent didn't make explicitly.

Ollama must be running locally:
    ollama serve   (in a separate terminal if not already running)
    ollama pull deepseek-r1:1.5b

Usage:
    python -m core.main distill --prompts math_prompts.jsonl
    (model defaults to deepseek-r1:1.5b via Ollama)
"""

import json
import time
import re
import random
import requests
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Iterator, Optional

from core.atoms  import Level, text_to_sentences
from core.reward import node_reward


# ─────────────────────────────────────────────
# OLLAMA CONFIG
# ─────────────────────────────────────────────

OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT  = 120   # seconds per request


# ─────────────────────────────────────────────
# DISTILL CONFIG
# ─────────────────────────────────────────────

@dataclass
class DistillConfig:
    """
    Configuration for the distillation agent.

    model_name       : Ollama model tag (default: deepseek-r1:1.5b)
    prompts_path     : JSONL file of prompts
    reward_threshold : above → positive ingestion, below → contrastive
    batch_size       : prompts per batch
    max_new_tokens   : max tokens per completion
    temperature      : generation temperature
    verbose          : print per-completion scoring
    """
    model_name       : str   = "deepseek-r1:1.5b"
    prompts_path     : str   = ""
    reward_threshold : float = 0.0
    batch_size       : int   = 4
    max_new_tokens   : int   = 512
    temperature      : float = 0.7
    verbose          : bool  = True


# ─────────────────────────────────────────────
# MATH CURRICULUM
# Structured topics for the agent to reason through
# ─────────────────────────────────────────────

MATH_CURRICULUM = [
    # Calculus
    "Explain step by step how to find the derivative of x squared using the power rule.",
    "Walk through the proof that the derivative of sin(x) is cos(x).",
    "Explain the chain rule for differentiation with a worked example.",
    "Prove the product rule for derivatives from first principles.",
    "Walk through computing the integral of x squared step by step.",
    "Explain what a limit is and how to compute lim(x->0) sin(x)/x.",
    "Prove that the derivative of e^x is e^x.",
    "Explain the fundamental theorem of calculus with a clear example.",
    "Walk through finding the critical points of f(x) = x^3 - 3x.",
    "Explain L'Hopital's rule and when to use it.",

    # Algebra
    "Prove the quadratic formula by completing the square.",
    "Explain how to solve a system of two linear equations step by step.",
    "Walk through factoring x^2 - 5x + 6 completely.",
    "Prove that the sum of first n integers equals n(n+1)/2.",
    "Explain logarithm properties with examples: log(ab), log(a/b), log(a^n).",
    "Walk through solving 2^x = 8 using logarithms.",
    "Prove that if a divides b and b divides c then a divides c.",
    "Explain the binomial theorem with a worked example.",

    # Number theory
    "Prove there are infinitely many prime numbers.",
    "Explain the Euclidean algorithm for finding GCD with an example.",
    "Prove that the square root of 2 is irrational.",
    "Explain Fermat's little theorem with a numerical example.",
    "Walk through the sieve of Eratosthenes to find primes up to 30.",
    "Prove that every integer greater than 1 has a prime factorization.",

    # Proofs and logic
    "Prove by induction that 2^n > n for all positive integers n.",
    "Explain proof by contradiction with a clear example.",
    "Prove that there are no integer solutions to x^2 + y^2 = 3 mod 4.",
    "Walk through a direct proof that the sum of two even numbers is even.",
    "Explain mathematical induction and prove the sum of odd numbers is n^2.",

    # Geometry
    "Prove the Pythagorean theorem using area.",
    "Explain why the sum of angles in a triangle is 180 degrees.",
    "Prove that the area of a circle is pi*r^2.",
    "Walk through finding the distance between two points using coordinates.",

    # Statistics and probability
    "Explain the central limit theorem intuitively.",
    "Walk through computing the expected value of rolling a fair die.",
    "Explain Bayes theorem with a concrete medical test example.",
    "Prove that variance is always non-negative.",
]


# ─────────────────────────────────────────────
# SYSTEM PROMPT — gives the agent its persona
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a mathematical reasoning agent. Your job is to think through mathematical problems carefully and step by step. 

When given a mathematical topic:
- Reason through it explicitly, showing every step
- State definitions clearly before using them
- Explain WHY each step follows from the previous
- Use precise mathematical language
- Think out loud — your reasoning process is what matters, not just the answer

Your responses become training material for a mathematical knowledge graph. Every sentence you write should be a clear, standalone mathematical statement that connects logically to the next."""


# ─────────────────────────────────────────────
# PROMPT READER
# ─────────────────────────────────────────────

def read_prompts(path: str) -> list[str]:
    """
    Read prompts from a JSONL file.
    Falls back to built-in curriculum if file is empty or missing.
    """
    prompts = []

    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        prompt = obj
                    elif isinstance(obj, dict):
                        prompt = (obj.get("prompt") or
                                  obj.get("text") or
                                  obj.get("question") or "")
                    else:
                        continue
                    prompt = str(prompt).strip()
                    if len(prompt) >= 10:
                        prompts.append(prompt)
                except json.JSONDecodeError:
                    if len(line) >= 10:
                        prompts.append(line)

    if not prompts:
        prompts = MATH_CURRICULUM.copy()

    return prompts


# ─────────────────────────────────────────────
# REWARD SCORING
# ─────────────────────────────────────────────

def score_completion(text: str) -> float:
    """
    Score a completion using the graph's reward function.
    Splits into sentences and averages node reward.
    Returns signed float — positive = coherent and novel.
    """
    sentences = text_to_sentences(text)
    if not sentences:
        return -1.0

    rewards    = []
    seen_words : set = set()

    for sent in sentences:
        words = sent.split()
        if len(words) < 3:
            continue
        r = node_reward(
            sent,
            Level.SENTENCE,
            seen_words = seen_words,
            context    = [],
        )
        rewards.append(r)
        seen_words |= set(re.findall(r'\b\w+\b', sent.lower()))

    if not rewards:
        return -1.0

    return round(float(sum(rewards) / len(rewards)), 4)


# ─────────────────────────────────────────────
# EXTRACT REASONING
# Pull out the most useful content from model output
# ─────────────────────────────────────────────

def extract_reasoning(text: str) -> str:
    """
    Clean up the combined thinking + content text.
    Remove any residual <think> tags if present.
    """
    text = re.sub(r'<think>|</think>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────
# OLLAMA GENERATION
# ─────────────────────────────────────────────

def generate_ollama(
    prompt      : str,
    model       : str,
    system      : str,
    temperature : float,
    max_tokens  : int,
) -> str:
    """
    Generate a completion via Ollama chat API.
    Returns the completion text or empty string on failure.
    """
    payload = {
        "model"    : model,
        "messages" : [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream"  : False,
        "options" : {
            "temperature"    : temperature,
            "num_predict"    : max_tokens,
            "top_p"          : 0.9,
            "repeat_penalty" : 1.1,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json    = payload,
            timeout = OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data    = response.json()
        message = data.get("message", {})

        # DeepSeek-R1 returns thinking separately from content
        # thinking = raw step-by-step reasoning (primary graph material)
        # content  = formatted final answer (secondary)
        thinking = message.get("thinking", "").strip()
        content  = message.get("content",  "").strip()

        if thinking and content:
            return thinking + "\n\n" + content
        return (thinking or content)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. "
            "Make sure Ollama is running: ollama serve"
        )
    except Exception as e:
        return ""


def format_prompt(system: str, user: str) -> str:
    """
    Format prompt for DeepSeek-R1 via Ollama.
    DeepSeek-R1 uses a simple concatenation — no special tokens needed
    when going through Ollama's API directly.
    """
    return f"{system}\n\n{user}"


# ─────────────────────────────────────────────
# DISTILLER — AGENT LOOP
# ─────────────────────────────────────────────

class Distiller:
    """
    Ollama-based reasoning agent for graph distillation.

    Usage:
        config    = DistillConfig(model_name="deepseek-r1:1.5b", ...)
        distiller = Distiller(config)
        for positive_texts, negative_texts in distiller.generate_batches():
            ingest_distillation_batch(positive_texts, negative_texts, ...)
    """

    def __init__(self, config: DistillConfig):
        self.config = config
        self._verify_ollama()

    def _verify_ollama(self):
        """Check that Ollama is running and the model is available."""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]

            # Check if our model is available (match by prefix)
            model_base = self.config.model_name.split(":")[0]
            available  = any(model_base in m for m in models)

            if not available:
                print(f"  Warning: '{self.config.model_name}' not found in Ollama.")
                print(f"  Available: {models}")
                print(f"  Run: ollama pull {self.config.model_name}")

            if self.config.verbose:
                print(f"\nOllama connected.")
                print(f"  Model:    {self.config.model_name}")
                print(f"  Endpoint: {OLLAMA_URL}")
                if available:
                    print(f"  Status:   model ready\n")

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "\nCannot connect to Ollama at localhost:11434.\n"
                "Start Ollama first:\n"
                "  1. Open a new terminal\n"
                "  2. Run: ollama serve\n"
                "  3. Then re-run distillation\n"
            )

    def generate_batches(self) -> Iterator[tuple[list[str], list[str]]]:
        """
        Main agent loop.

        Reads prompts, generates reasoning chains via Ollama,
        scores each chain, splits into positive/negative,
        yields per batch for ingestion.
        """
        prompts    = read_prompts(self.config.prompts_path)
        batch_size = self.config.batch_size

        if self.config.verbose:
            print(f"Starting agent loop.")
            print(f"  Prompts:   {len(prompts)}")
            print(f"  Batch:     {batch_size}")
            print(f"  Threshold: {self.config.reward_threshold}\n")

        total_pos   = 0
        total_neg   = 0
        batch_num   = 0
        prompt_buf  : list[str] = []

        # Shuffle so the agent covers topics in varied order
        random.shuffle(prompts)

        for prompt in prompts:
            prompt_buf.append(prompt)

            if len(prompt_buf) < batch_size:
                continue

            batch_num += 1
            positive, negative = self._process_batch(prompt_buf, batch_num)
            total_pos += len(positive)
            total_neg += len(negative)

            yield positive, negative
            prompt_buf = []

        # Flush remaining
        if prompt_buf:
            batch_num += 1
            positive, negative = self._process_batch(prompt_buf, batch_num)
            total_pos += len(positive)
            total_neg += len(negative)
            yield positive, negative

        if self.config.verbose:
            print(f"\nAgent loop complete.")
            print(f"  Batches:  {batch_num}")
            print(f"  Positive: {total_pos}")
            print(f"  Negative: {total_neg}")

    def _process_batch(
        self,
        prompts   : list[str],
        batch_num : int,
    ) -> tuple[list[str], list[str]]:
        """
        Generate, extract reasoning, score, and split one batch.
        """
        positive : list[str] = []
        negative : list[str] = []

        for prompt in prompts:
            # Generate via Ollama
            raw_output = generate_ollama(
                prompt      = prompt,
                model       = self.config.model_name,
                system      = SYSTEM_PROMPT,
                temperature = self.config.temperature,
                max_tokens  = self.config.max_new_tokens,
            )

            if not raw_output or len(raw_output.strip()) < 20:
                if self.config.verbose:
                    print(f"  [batch {batch_num}] Empty output for: {prompt[:50]}...")
                continue

            # Extract the reasoning content
            reasoning = extract_reasoning(raw_output)

            if not reasoning or len(reasoning.strip()) < 20:
                continue

            # Score the reasoning chain
            score = score_completion(reasoning)

            routing = "+" if score > self.config.reward_threshold else "-"

            if self.config.verbose:
                preview = reasoning[:80].replace('\n', ' ')
                print(f"  [{routing}] score={score:+.3f}  {preview}...")

            if score > self.config.reward_threshold:
                positive.append(reasoning)
            else:
                negative.append(reasoning)

        return positive, negative

    def release(self):
        """No-op for Ollama — model lives in Ollama process, not here."""
        pass


# ─────────────────────────────────────────────
# PROMPT GENERATION UTILITIES
# ─────────────────────────────────────────────

def generate_math_prompts(output_path: str, n: int = 1000):
    """
    Generate a starter set of mathematical reasoning prompts.
    Saves to a JSONL file.
    """
    templates = MATH_CURRICULUM * (n // len(MATH_CURRICULUM) + 1)
    templates = templates[:n]

    with open(output_path, "w", encoding="utf-8") as f:
        for template in templates:
            f.write(json.dumps({"prompt": template}) + "\n")

    print(f"Generated {len(templates)} math prompts → {output_path}")


if __name__ == "__main__":
    print("=== distill.py test (Ollama connection check) ===\n")

    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama running. Models available: {models}")

        # Quick generation test
        result = generate_ollama(
            prompt      = "What is 2 + 2? Think step by step.",
            model       = "deepseek-r1:1.5b",
            temperature = 0.7,
            max_tokens  = 100,
        )
        print(f"\nTest generation:\n{result[:200]}")

        reasoning = extract_reasoning(result)
        score     = score_completion(reasoning)
        print(f"\nExtracted reasoning score: {score:+.3f}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")
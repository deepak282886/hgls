"""
distill.py

7B model distillation pipeline.

Queries a 7B parameter language model and uses its outputs as
corpus material for the graph — not as ground truth reasoning,
but as a source of fluent, domain-spanning text that the graph
then reorganises through its own reward function and collision loop.

The 7B model is used as a corpus generator, not as a teacher.
Hallucinations and incoherence enter the graph but receive low
reward and weak edges. Selection pressure handles them over time.

Hardware target: 8GB VRAM
    7B model in 4-bit NF4 quantization: ~4.0–4.5 GB VRAM
    Leaves ~3.5 GB for KV cache and activations.
    Batch size 4 at max_new_tokens=256 fits comfortably.
    Batch size 8 is possible but tight — use with short inputs.

Quantization:
    bitsandbytes NF4 with double quantization enabled.
    Compute dtype: float16.
    This is the standard setup for 7B on consumer GPUs.

Flow:
    1. Load prompts from JSONL file
    2. Generate completions in batches (batch_size=4 default)
    3. Score each completion with the reward function
    4. Split by threshold:
           score > threshold  → positive ingestion
           score ≤ threshold  → contrastive ingestion
    5. Yield (positive_texts, negative_texts) per batch
       for ingest_distillation_batch() in main.py

The Distiller keeps the model loaded across all batches.
Load once, generate everything, then release.

Supported model sources:
    HuggingFace hub: "mistralai/Mistral-7B-Instruct-v0.2"
    Local path:      "/models/mistral-7b"
    GGUF (future):   not supported — use transformers only

Recommended models for 8GB VRAM (4-bit):
    mistralai/Mistral-7B-Instruct-v0.2    (~4.1 GB)
    meta-llama/Meta-Llama-3-8B-Instruct   (~4.5 GB)
    microsoft/Phi-3-mini-4k-instruct      (~2.2 GB, fastest)
"""

import json
import time
import re
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Iterator, Optional

from core.atoms  import Level
from core.reward import node_reward
from core.atoms  import text_to_sentences


# ─────────────────────────────────────────────
# DISTILL CONFIG
# ─────────────────────────────────────────────

@dataclass
class DistillConfig:
    """
    Configuration for the distillation pipeline.

    model_name       : HuggingFace model ID or local path
    prompts_path     : JSONL file of prompts — one per line,
                       {"prompt": "..."} or {"text": "..."}
    reward_threshold : completions above this go positive,
                       at or below go contrastive (default 0.0)
    batch_size       : prompts per generation batch.
                       Keep at 4 for 8GB VRAM with 256 new tokens.
                       Drop to 2 if OOM errors occur.
    max_new_tokens   : max tokens per completion (default 256)
    temperature      : generation temperature (default 0.8)
                       Higher = more diverse but less coherent.
    top_p            : nucleus sampling cutoff (default 0.9)
    device_map       : "auto" lets accelerate assign layers.
                       On single GPU this puts everything on cuda:0.
    load_in_4bit     : enable 4-bit NF4 quantization (default True)
                       Set False only if you have >14GB VRAM.
    verbose          : print scoring and routing per batch
    """
    model_name       : str   = "mistralai/Mistral-7B-Instruct-v0.2"
    prompts_path     : str   = ""
    reward_threshold : float = 0.0
    batch_size       : int   = 4
    max_new_tokens   : int   = 256
    temperature      : float = 0.8
    top_p            : float = 0.9
    device_map       : str   = "auto"
    load_in_4bit     : bool  = True
    verbose          : bool  = True


# ─────────────────────────────────────────────
# PROMPT READER
# ─────────────────────────────────────────────

def read_prompts(path: str) -> Iterator[str]:
    """
    Read prompts from a JSONL file.

    Accepted formats per line:
        {"prompt": "..."}
        {"text": "..."}
        {"question": "..."}
        "raw string"

    Prompts that are too short (<10 chars) are skipped.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

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
                    prompt = (
                        obj.get("prompt") or
                        obj.get("text")   or
                        obj.get("question") or
                        ""
                    )
                else:
                    continue

                prompt = str(prompt).strip()
                if len(prompt) >= 10:
                    yield prompt

            except json.JSONDecodeError:
                # Treat line as raw text prompt
                if len(line) >= 10:
                    yield line


# ─────────────────────────────────────────────
# REWARD SCORING
# ─────────────────────────────────────────────

def score_completion(text: str) -> float:
    """
    Score a model completion using the graph's reward function.

    Splits text into sentences and computes the average node
    reward across all sentences. This gives a stable signal
    even for short completions.

    Returns a signed float — positive = coherent and novel,
    negative = incoherent or repetitive.

    Uses an empty seen_words set — no prior context assumed.
    The graph will provide real context when the text is ingested.
    """
    sentences = text_to_sentences(text)
    if not sentences:
        return -1.0

    rewards = []
    seen_words: set = set()

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
# DISTILLER
# ─────────────────────────────────────────────

class Distiller:
    """
    Loads a 7B model once and generates completions in batches.

    Usage:
        config   = DistillConfig(model_name="...", prompts_path="...")
        distiller = Distiller(config)
        for positive_texts, negative_texts in distiller.generate_batches():
            ingest_distillation_batch(positive_texts, negative_texts, ...)
    """

    def __init__(self, config: DistillConfig):
        self.config    = config
        self.model     = None
        self.tokenizer = None
        self.pipeline  = None
        self._load_model()

    def _load_model(self):
        """
        Load the model with 4-bit NF4 quantization for 8GB VRAM.

        Memory breakdown at 4-bit:
            Mistral 7B  : ~4.1 GB model weights
            KV cache    : ~1.5 GB at batch=4, seq=512
            Activations : ~0.5 GB
            Total       : ~6.1 GB — fits in 8GB with headroom

        If you see CUDA OOM:
            - Reduce batch_size to 2
            - Reduce max_new_tokens to 128
            - Or switch to Phi-3-mini (~2.2 GB in 4-bit)
        """
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
                pipeline as hf_pipeline,
            )
        except ImportError:
            raise ImportError(
                "transformers and bitsandbytes required for distillation.\n"
                "Install with: pip install transformers bitsandbytes accelerate"
            )

        if self.config.verbose:
            print(f"\nLoading model: {self.config.model_name}")
            print(f"  Quantization: {'4-bit NF4' if self.config.load_in_4bit else 'fp16'}")
            print(f"  Device map:   {self.config.device_map}")

        t0 = time.time()

        # ── Quantization config ──────────────
        if self.config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_quant_type       = "nf4",        # NF4 is best for LLMs
                bnb_4bit_compute_dtype    = torch.float16, # compute in fp16
                bnb_4bit_use_double_quant = True,          # saves ~0.4 GB extra
            )
        else:
            quant_config = None

        # ── Tokenizer ────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side = "left",   # left-pad for batch generation
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Model ────────────────────────────
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config = quant_config,
            device_map          = self.config.device_map,
            torch_dtype         = torch.float16,
            trust_remote_code   = True,   # needed for some models (Phi-3)
        )
        self.model.eval()

        # ── Pipeline ─────────────────────────
        # Using HF pipeline for clean batched generation
        self.pipeline = hf_pipeline(
            "text-generation",
            model     = self.model,
            tokenizer = self.tokenizer,
        )

        elapsed = time.time() - t0
        if self.config.verbose:
            print(f"  Model loaded in {elapsed:.1f}s\n")

    def _format_prompt(self, prompt: str) -> str:
        """
        Format a raw prompt into the model's chat template if available.

        Falls back to raw prompt if the tokenizer has no chat template.
        Mistral and LLaMA 3 both have chat templates — this ensures
        the instruction format is correct for instruct models.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize        = False,
                    add_generation_prompt = True,
                )
            except Exception:
                pass
        return prompt

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Generate completions for a batch of prompts.

        Returns list of completion strings (prompt stripped out).

        Generation parameters:
            temperature = config.temperature (default 0.8)
                          Lower for more consistent quality,
                          higher for more diverse corpus coverage.
            top_p       = 0.9 (nucleus sampling)
            do_sample   = True (sampling, not greedy)
            batch_size  = config.batch_size

        On 8GB VRAM with 4-bit Mistral 7B:
            batch_size=4, max_new_tokens=256 → ~2-3 seconds per batch
            batch_size=4, max_new_tokens=512 → ~4-6 seconds per batch
        """
        formatted = [self._format_prompt(p) for p in prompts]

        outputs = self.pipeline(
            formatted,
            max_new_tokens = self.config.max_new_tokens,
            temperature    = self.config.temperature,
            top_p          = self.config.top_p,
            do_sample      = True,
            pad_token_id   = self.tokenizer.pad_token_id,
            batch_size     = self.config.batch_size,
            return_full_text = False,   # only return the new tokens
        )

        completions = []
        for out in outputs:
            if isinstance(out, list):
                text = out[0].get("generated_text", "")
            else:
                text = out.get("generated_text", "")
            completions.append(text.strip())

        return completions

    def generate_batches(self) -> Iterator[tuple[list[str], list[str]]]:
        """
        Main generation loop.

        Reads prompts, generates in batches, scores each completion,
        splits into positive and negative, yields per batch.

        Yields:
            (positive_texts, negative_texts)
            where positive_texts are completions with score > threshold
            and   negative_texts are completions with score ≤ threshold

        The caller (main.py cmd_distill) passes these to
        ingest_distillation_batch().
        """
        prompts_buffer: list[str] = []
        batch_num     = 0
        total_prompts = 0
        total_pos     = 0
        total_neg     = 0

        for prompt in read_prompts(self.config.prompts_path):
            prompts_buffer.append(prompt)
            total_prompts += 1

            if len(prompts_buffer) < self.config.batch_size:
                continue

            # Process batch
            batch_num += 1
            positive, negative = self._process_batch(
                prompts_buffer, batch_num
            )
            total_pos += len(positive)
            total_neg += len(negative)

            if self.config.verbose:
                print(
                    f"  Batch {batch_num:4d}  "
                    f"prompts={len(prompts_buffer)}  "
                    f"positive={len(positive)}  "
                    f"negative={len(negative)}"
                )

            yield positive, negative
            prompts_buffer = []

        # Flush remaining prompts
        if prompts_buffer:
            batch_num += 1
            positive, negative = self._process_batch(
                prompts_buffer, batch_num
            )
            total_pos += len(positive)
            total_neg += len(negative)

            if self.config.verbose:
                print(
                    f"  Batch {batch_num:4d} (final)  "
                    f"prompts={len(prompts_buffer)}  "
                    f"positive={len(positive)}  "
                    f"negative={len(negative)}"
                )

            yield positive, negative

        if self.config.verbose:
            print(f"\nGeneration complete.")
            print(f"  Total prompts : {total_prompts}")
            print(f"  Total batches : {batch_num}")
            print(f"  Positive      : {total_pos}")
            print(f"  Negative      : {total_neg}")
            if total_prompts > 0:
                pos_rate = total_pos / total_prompts
                print(f"  Positive rate : {pos_rate:.1%}")

    def _process_batch(
        self,
        prompts   : list[str],
        batch_num : int,
    ) -> tuple[list[str], list[str]]:
        """
        Generate, score, and split one batch.

        Returns (positive_texts, negative_texts).
        """
        try:
            completions = self._generate_batch(prompts)
        except Exception as e:
            if self.config.verbose:
                print(f"  Generation error in batch {batch_num}: {e}")
            return [], []

        positive: list[str] = []
        negative: list[str] = []

        for prompt, completion in zip(prompts, completions):
            if not completion or len(completion.strip()) < 10:
                continue

            # Score the completion (not the prompt+completion combined —
            # we want to know if the output itself is good)
            score = score_completion(completion)

            if self.config.verbose:
                preview = completion[:60].replace('\n', ' ')
                routing = "+" if score > self.config.reward_threshold else "-"
                print(f"    [{routing}] score={score:+.3f}  {preview}...")

            if score > self.config.reward_threshold:
                # Positive: ingest with sequential edges
                positive.append(completion)
            else:
                # Negative: ingest with contrastive edges
                negative.append(completion)

        return positive, negative

    def release(self):
        """
        Release model from GPU memory.

        Call when distillation is complete and you want to free
        VRAM for other processes (e.g., if running inference tests).
        """
        try:
            import torch
            if self.model is not None:
                del self.model
                self.model = None
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            torch.cuda.empty_cache()
            if self.config.verbose:
                print("Model released from GPU memory.")
        except Exception:
            pass


# ─────────────────────────────────────────────
# PROMPT GENERATION UTILITIES
# ─────────────────────────────────────────────

def generate_math_prompts(output_path: str, n: int = 1000):
    """
    Generate a starter set of mathematical reasoning prompts.

    Covers arithmetic, algebra, calculus, number theory, proofs.
    Saves to a JSONL file that can be passed to --prompts.

    Usage:
        from core.distill import generate_math_prompts
        generate_math_prompts("math_prompts.jsonl", n=5000)
    """
    templates = [
        # Arithmetic
        "Explain how to add fractions with different denominators.",
        "What is the difference between a prime number and a composite number?",
        "Prove that the square root of 2 is irrational.",
        "Explain the distributive property of multiplication over addition.",
        "What is modular arithmetic and how is it used?",
        # Algebra
        "How do you solve a system of two linear equations?",
        "What is the quadratic formula and how is it derived?",
        "Explain what a function is in mathematics.",
        "What is the difference between permutations and combinations?",
        "How do logarithms relate to exponentials?",
        # Calculus
        "What is the definition of a derivative?",
        "Explain the chain rule for differentiation.",
        "What is the fundamental theorem of calculus?",
        "How do you compute the area under a curve?",
        "What is L'Hôpital's rule and when is it used?",
        # Number theory
        "State and prove Euclid's theorem on the infinitude of primes.",
        "What is the Euclidean algorithm for finding GCD?",
        "Explain Fermat's little theorem.",
        "What are perfect numbers? Give an example.",
        "How does the sieve of Eratosthenes work?",
        # Proofs
        "Prove that the sum of the first n natural numbers is n(n+1)/2.",
        "Prove by induction that 2^n > n for all positive integers n.",
        "What is a proof by contradiction? Give an example.",
        "Prove that there are infinitely many prime numbers.",
        "What is mathematical induction and how does it work?",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        while count < n:
            for template in templates:
                if count >= n:
                    break
                f.write(json.dumps({"prompt": template}) + "\n")
                count += 1

    print(f"Generated {count} math prompts → {output_path}")


if __name__ == "__main__":
    print("=== distill.py utility test (no model load) ===\n")

    # Test prompt reading
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_path = os.path.join(tmpdir, "test_prompts.jsonl")
        with open(prompts_path, "w") as f:
            f.write(json.dumps({"prompt": "What is the derivative of x squared?"}) + "\n")
            f.write(json.dumps({"prompt": "Prove the Pythagorean theorem."}) + "\n")
            f.write(json.dumps({"text": "Explain Euclidean geometry."}) + "\n")
            f.write('"What is a prime number?"\n')

        prompts = list(read_prompts(prompts_path))
        print(f"Read {len(prompts)} prompts:")
        for p in prompts:
            print(f"  - {p}")

        # Test reward scoring
        print("\nReward scoring test:")
        test_completions = [
            "The derivative of x squared is 2x. We apply the power rule: "
            "bring down the exponent and reduce it by one.",
            "banana banana banana banana banana banana banana banana banana",
            "The proof proceeds by contradiction. Assume there are finitely "
            "many primes. Multiply them all together and add one.",
        ]
        for text in test_completions:
            score = score_completion(text)
            routing = "positive" if score > 0.0 else "contrastive"
            print(f"  score={score:+.3f}  [{routing}]  {text[:50]}...")

        print("\n(Full distillation requires transformers + bitsandbytes + GPU)")
        print("Install: pip install transformers bitsandbytes accelerate")

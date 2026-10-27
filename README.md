checked for correctness** and **timed on real NVIDIA H200 GPUs**, so you get honest feedback without owning a GPU.
- The baseline (SGLang) is strong, so even a small speedup teaches you a lot about what actually makes GPU code fast.

---

## The two tracks

| Track | Dates (PT) | What you do | Hardware |
|---|---|---|---|
| **Kernel** | Now → **Nov 4, 2026, 11:59 pm** | Optimize five individual kernels and beat SGLang's version of each | 1 × H200 |
| **Serving** | **Nov 5 – Nov 19, 2026** | Patch SGLang itself to serve the full model faster under an agentic-coding workload (1 to 32 concurrent users) | 8 × H200 |

The **Kernel track** is about making single operations fast in isolation.
The **Serving track** is about making the whole system fast end-to-end: scheduling, batching, memory, and how all the kernels work together.

**This repository focuses on the Kernel track**, which is the more approachable entry point for a beginner.

---

## The five problems (Kernel track)

Each problem is one piece of the model's forward pass. Listed roughly from **most approachable** to **most advanced**:

### 1. `muse_bf16_proj` — Dense BF16 projections ← *my focus*
The model constantly multiplies its activations by large weight matrices ("projections"), for example when computing attention queries/keys/values or the router scores that decide which experts handle each token. This problem is about doing those **matrix multiplications in BF16** as fast as possible.

*What you learn:* GEMM vs GEMV, memory-bound vs compute-bound workloads, tiling, why small batch sizes behave so differently from large ones, and when a library call (cuBLAS) is already the best answer.

### 2. `muse_attention_decode` — Gated attention decode over FP8 KV cache
During generation, each new token "looks back" at all previous tokens through **attention**. Their keys and values are stored in a **KV cache** in compressed **FP8** format. This kernel reads that cache and computes attention for the new token, with an extra gating step.

*What you learn:* attention math, memory bandwidth limits, low-precision formats, splitting long sequences across many thread blocks.

### 3. `muse_moe_w4a16` — Routed Mixture-of-Experts with 4-bit weights
In an MoE layer, each token is sent to only a few of many "expert" networks. The expert weights are stored in **NVFP4** (4-bit) and must be dequantized on the fly while multiplying with 16-bit activations.

*What you learn:* quantization and dequantization, grouped/batched GEMMs, load balancing when some experts get many tokens and others get none.

### 4. `muse_decode_layer` — A full decoder layer (decode)
Combines the pieces above into **one complete transformer layer** for the token-by-token generation phase.

*What you learn:* kernel fusion, reducing launches and memory round-trips, thinking about the whole layer instead of one operation.

### 5. `muse_prefill_layer` — A full layer (prefill)
The same idea, but for **prefill**: processing the whole input prompt at once. Shapes are much larger, so the bottlenecks shift from memory toward raw compute.

*What you learn:* how optimization strategy changes between prefill (compute-heavy) and decode (memory-heavy).

---

## How a submission is judged

1. **Correctness first.** Your output is compared against a reference implementation. If it is wrong, it does not count, no matter how fast it is.
2. **Then speed.** The kernel is timed across several input shapes on an H200 and compared to SGLang's baseline. The result is reported as a **speedup** (above 1.0× means faster than SGLang).

**My definition of success for this project:** a submission that **passes correctness**. Any speedup is a bonus.

---

## Mini-glossary for beginners

| Term | Meaning |
|---|---|
| **Kernel** | A function that runs on the GPU, executed by thousands of threads in parallel |
| **GEMM / GEMV** | General Matrix-Matrix / Matrix-Vector multiply. GEMV is the case with a single row (one token) |
| **BF16, FP8, NVFP4** | Number formats with 16, 8 and 4 bits. Fewer bits = less memory to move, but less precision |
| **Memory-bound** | Speed is limited by how fast data is read from memory, not by arithmetic |
| **Compute-bound** | Speed is limited by how fast the GPU can do math |
| **Prefill / Decode** | Processing the whole prompt at once / generating one new token at a time |
| **KV cache** | Stored keys and values of past tokens, reused at every decode step |
| **Mixture-of-Experts (MoE)** | A layer with many sub-networks ("experts"); each token only uses a few |
| **SGLang** | An open-source LLM serving engine; its kernels are the baseline to beat |
| **H200** | The NVIDIA Hopper data-center GPU used for judging |
| **Speedup** | Baseline time ÷ your time. 1.10× means 10% faster |

---

## Repository layout

Files created by `okc init` for each problem:

| File | Purpose |
|---|---|
| `submission.py` | The kernel I edit and submit |
| `baseline.py` | SGLang's implementation (what I'm trying to beat) |
| `reference.py` | The "ground truth" used for correctness checking |
| `TASK.md` | Official description of the problem |
| `AGENTS.md` | Official rules and guidance for AI coding agents |

---

## Reproducing my workflow

```bash
# 1. Install and log in to the challenge CLI (one time)
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
export PATH="$HOME/.okc/bin:$PATH"
okc login

# 2. Download a problem
okc init                     # choose muse_bf16_proj

# 3. Test a kernel on the remote H200 (limited runs per day)
okc check ...                # see `okc --help` for exact syntax

# 4. Submit the best passing version
okc submit muse_bf16_proj/submission.py
```

I develop on Windows + WSL with VS Code and use **Claude Code** as an AI pair-programmer: it helps me read the problem, explain the math, and draft kernel variants, while I make the decisions and record what I learn.

---

## Learning journal

Every attempt is recorded here, including the ones that got slower. Failures teach as much as successes.

| Version | Idea | Correct? | Speedup | What I learned |
|---|---|---|---|---|
| v001 | Unmodified starter | — | — | — |

---

## Disclaimer & license

This is an independent student project and is **not affiliated with** Meta, the challenge organizers, or SGLang. For official information always refer to [openkernelchallenge.dev](https://openkernelchallenge.dev/) and its rules page.

Code is released under the **Apache-2.0** license, matching the challenge's submission terms.
# Problem overview: `muse_bf16_proj`

Sources: `muse_bf16_proj/TASK.md`, `AGENTS.md`, `reference.py`, `baseline.py`, `submission.py`.
Hardware: 1× H200. Peak numbers the judge uses: **4.8 TB/s** HBM bandwidth and **989 TFLOP/s** dense BF16.

## 1. What is computed

Every case is one matrix multiply:

```
out[m, n] = x[m, k] @ W[n, k].T
```

- `x` = **activations**: one row per token, BF16. New every call.
- `W` = **weights** of one layer of the Muse Spark Open model. Given once in `prepare()` (untimed).
- Output is BF16, except the **router**, which has FP32 weights and returns FP32.

These are the "dense projections" of one transformer layer, as one GPU of an 8-GPU server sees
them. **TP8** (tensor parallel over 8 GPUs) means each big weight is cut into 8 slices and every
GPU holds one, so the shapes below are per-GPU slices, not the full model.

### The five layers (weights)

| layer | W shape [N, K] | dtype | weight size | what it does |
|---|---|---|---|---|
| `qkv_gate` | 2304 × 8192 | BF16 | 36 MiB | hidden (8192) → query, key, value and gate for attention, fused into one matmul |
| `o` | 8192 × 1024 | BF16 | 16 MiB | attention output (1024 on this GPU) → back to hidden size |
| `pooler` | 4096 × 8192 | BF16 | 64 MiB | hidden → MoE width |
| `unpooler` | 8192 × 4096 | BF16 | 64 MiB (8 MiB when sliced) | MoE width → hidden. **At m ≤ 8 only the first 1024 rows are used** (N = 1024): this GPU computes its 1/8 slice and SGLang would all-gather afterwards (not timed). Above 8 tokens the whole weight is used. |
| `router` | 256 × 8192 | **FP32** | 8 MiB | hidden → 256 expert scores (picks which MoE experts each token goes to) |

Terms:
- **BF16** (bfloat16): a 2-byte float with FP32's range but only ~3 significant decimal digits.
- **Hidden size**: the width of the vector that represents each token between layers (8192 here).
- **MoE** (mixture of experts): a layer where each token is sent to a few of many small "expert" networks; the router decides which.

### What m means: decode vs verify

`m` = number of token rows in `x` this step.

- **Decode** loads (m = 1, 16, 32): normal generation, one new token per request, so m = number of
  requests in the batch (batch size 1, 16, 32).
- **Verify** loads (m = 8, 128, 256): **speculative decoding**. A cheap draft model guesses 8
  tokens per request, and the big model checks all 8 in one step. So m = 8 × requests:
  "verify 1" = 1 request = m 8, "verify 16" = m 128, "verify 32" = m 256.

## 2. GEMV vs GEMM

- **GEMV** (matrix × vector): x has a single row (m = 1). Each weight is read once and used for
  exactly one multiply-add.
- **GEMM** (matrix × matrix): x has many rows, so each weight that's loaded gets reused m times.

In between is the **skinny GEMM** (small m like 8–32): technically a GEMM, but it behaves like a
GEMV because each weight is still only reused a handful of times.

## 3. Memory-bound or compute-bound? (the roofline)

Two speed limits:
- **Memory time** = bytes moved / 4.8 TB/s.
- **Compute time** = FLOPs / 989 TFLOP/s, where FLOPs = 2·m·n·k (one multiply + one add per term).

Whichever is larger is the floor. **Arithmetic intensity** (AI) = FLOPs / bytes. The
**ridge point** where both limits are equal is 989e12 / 4.8e12 ≈ **206 FLOP/byte**.
Below that, a kernel is **memory-bound** (waiting on data); above it, **compute-bound** (waiting on math).

The bytes are almost all **weights**. Example, `pooler` at m = 1:
weights 4096 × 8192 × 2 B = 67 MB, x only 16 KB, out 8 KB.
FLOPs = 2 × 1 × 4096 × 8192 = 67 MFLOP. AI ≈ 1 FLOP/byte, 200× below the ridge.
Moving 67 MB takes 14 µs; the math alone would take 0.07 µs. **The GPU's math units sit idle ~99.5% of the time.**

Because weight bytes are fixed and FLOPs grow with m, **AI ≈ m** for BF16 layers (≈ m/2 for the
FP32 router). So the regime is set almost entirely by m.

### All 30 scored shapes

(Numbers computed with the same formula as `speed_of_light_ns` in reference.py; they match the "floor" column in TASK.md.)

| load | m | layer | N × K | kind | AI (FLOP/B) | mem µs | compute µs | bound |
|---|---|---|---|---|---|---|---|---|
| decode 1 | 1 | qkv_gate | 2304×8192 | GEMV | 1.0 | 7.87 | 0.04 | memory |
| decode 1 | 1 | o | 8192×1024 | GEMV | 1.0 | 3.50 | 0.02 | memory |
| decode 1 | 1 | pooler | 4096×8192 | GEMV | 1.0 | 13.99 | 0.07 | memory |
| decode 1 | 1 | unpooler | **1024**×4096 | GEMV | 1.0 | 1.75 | 0.01 | memory |
| decode 1 | 1 | router (FP32) | 256×8192 | GEMV | 0.5 | 1.75 | 0.00 | memory |
| verify 1 | 8 | qkv_gate | 2304×8192 | skinny GEMM | 8.0 | 7.90 | 0.31 | memory |
| verify 1 | 8 | o | 8192×1024 | skinny GEMM | 7.9 | 3.53 | 0.14 | memory |
| verify 1 | 8 | pooler | 4096×8192 | skinny GEMM | 8.0 | 14.02 | 0.54 | memory |
| verify 1 | 8 | unpooler | **1024**×4096 | skinny GEMM | 7.9 | 1.76 | 0.07 | memory |
| verify 1 | 8 | router (FP32) | 256×8192 | skinny GEMM | 3.9 | 1.78 | 0.03 | memory |
| decode 16 | 16 | qkv_gate | 2304×8192 | skinny GEMM | 15.9 | 7.93 | 0.61 | memory |
| decode 16 | 16 | o | 8192×1024 | skinny GEMM | 15.7 | 3.56 | 0.27 | memory |
| decode 16 | 16 | pooler | 4096×8192 | skinny GEMM | 15.9 | 14.06 | 1.09 | memory |
| decode 16 | 16 | unpooler | 8192×4096 | skinny GEMM | 15.9 | 14.06 | 1.09 | memory |
| decode 16 | 16 | router (hi+lo) | 256×8192 | skinny GEMM | 7.7 | 1.81 | 0.07 | memory |
| decode 32 | 32 | qkv_gate | 2304×8192 | skinny GEMM | 31.4 | 8.00 | 1.22 | memory |
| decode 32 | 32 | o | 8192×1024 | skinny GEMM | 30.9 | 3.62 | 0.54 | memory |
| decode 32 | 32 | pooler | 4096×8192 | skinny GEMM | 31.6 | 14.14 | 2.17 | memory |
| decode 32 | 32 | unpooler | 8192×4096 | skinny GEMM | 31.6 | 14.14 | 2.17 | memory |
| decode 32 | 32 | router (hi+lo) | 256×8192 | skinny GEMM | 15.0 | 1.86 | 0.14 | memory |
| verify 16 | 128 | qkv_gate | 2304×8192 | GEMM | 119.5 | 8.42 | 4.89 | memory (math not negligible) |
| verify 16 | 128 | o | 8192×1024 | GEMM | 112.2 | 3.99 | 2.17 | memory (math not negligible) |
| verify 16 | 128 | pooler | 4096×8192 | GEMM | 122.3 | 14.64 | 8.69 | memory (math not negligible) |
| verify 16 | 128 | unpooler | 8192×4096 | GEMM | 122.3 | 14.64 | 8.69 | memory (math not negligible) |
| verify 16 | 128 | router (hi+lo) | 256×8192 | GEMM | 50.6 | 2.21 | 0.54 | memory |
| verify 32 | 256 | qkv_gate | 2304×8192 | GEMM | 224.1 | 8.98 | 9.77 | **compute** (≈ balanced) |
| verify 32 | 256 | o | 8192×1024 | GEMM | 199.8 | 4.48 | 4.34 | balanced (memory by a hair) |
| verify 32 | 256 | pooler | 4096×8192 | GEMM | 234.1 | 15.29 | 17.37 | **compute** (≈ balanced) |
| verify 32 | 256 | unpooler | 8192×4096 | GEMM | 234.1 | 15.29 | 17.37 | **compute** (≈ balanced) |
| verify 32 | 256 | router (hi+lo) | 256×8192 | GEMM | 83.6 | 2.68 | 1.09* | memory |

\* The router's hi + lo trick does two BF16 matmuls, so its real math is 2× this, still well under the memory time.

**Summary:**
- **m = 1 → GEMV, extremely memory-bound.** Job: stream the weights at full bandwidth, nothing else matters.
- **m = 8, 16, 32 → skinny GEMM, still clearly memory-bound** (AI 8–32 vs ridge 206). Same job as GEMV, but now the
  kernel must do the math for 8–32 rows without slowing the weight stream. Note the 16 and 32 rows cost almost
  the same as m = 1 (pooler: 14.0 → 14.1 µs): extra tokens are nearly free when you're memory-bound.
- **m = 128 → GEMM, memory-bound on paper**, but compute is ~60% of the memory time. Real GPUs reach maybe
  ~70–80% of their peak FLOP/s, so here the math must be efficient (tensor cores, good tiling) to stay hidden.
- **m = 256 → GEMM right at the ridge.** Both the weight stream and the tensor-core math must be near peak at
  the same time. This is where a well-tuned library like cuBLAS is hardest to beat.

Caveat: 4.8 TB/s and 989 TFLOP/s are datasheet peaks. Real kernels typically reach somewhat less (often quoted
around ~85–90% of peak bandwidth), so nobody hits the "floor" exactly.

Also: **the L2 cache is flushed before every timed call** (the judge reads a buffer bigger than L2). Weights
always come from HBM; you can't rely on them being cached from the previous call.

## 4. How the score works

- Each **load** (decode 1, decode 16, decode 32, verify 1, verify 16, verify 32) = its 5 shapes,
  each run 62 times per step (62 = one per model layer, presumably). Load time = sum of shape times × 62.
- Speedup per load = SGLang time / your time. **Final score = geometric mean over the 6 loads**, so each load counts equally.
- Inside a load, the big shapes dominate. Floor times per layer:

| load | qkv_gate | o | pooler | unpooler | router | total | pooler+unpooler share |
|---|---|---|---|---|---|---|---|
| decode 1 (m 1) | 7.9 | 3.5 | 14.0 | 1.7 | 1.8 | 28.9 µs | 54% |
| verify 1 (m 8) | 7.9 | 3.5 | 14.0 | 1.8 | 1.8 | 29.0 µs | 55% |
| decode 16 | 7.9 | 3.6 | 14.1 | 14.1 | 1.8 | 41.5 µs | 68% |
| decode 32 | 8.0 | 3.6 | 14.1 | 14.1 | 1.9 | 41.7 µs | 68% |
| verify 16 (m 128) | 8.4 | 4.0 | 14.6 | 14.6 | 2.2 | 43.8 µs | 67% |
| verify 32 (m 256) | 9.8 | 4.5 | 17.4 | 17.4 | 2.7 | 51.8 µs | 67% |

  These are *floors*, not SGLang's real times. Tiny shapes (router, sliced unpooler) can still matter if
  SGLang is far from their floor there (e.g. fixed per-kernel overhead of a couple of µs on a 1.8 µs job).
  The real per-shape SGLang times appear only after an `okc check`.
- The judge subtracts the ~4 µs fixed cost of a CUDA graph replay from both sides.

## 5. What SGLang's baseline (`baseline.py`) does

`prepare()`: makes weights contiguous, and splits the FP32 router weight into two BF16 matrices:
`hi = bf16(W)` and `lo = bf16(W − hi)`, stored transposed. hi + lo recovers W to roughly 16 bits of mantissa.

`kernel_fn()`:
- **qkv_gate, o, pooler, unpooler** → `_bf16_gemm_dispatch_impl`, which on H200 calls **cuBLAS**
  (NVIDIA's closed-source, heavily tuned matmul library). One kernel per call. For the unpooler at m ≤ 8 it passes
  only the first 1024 rows.
- **router, m ≤ 8** → `F.linear(x.float(), W)`: converts x to FP32 (one small kernel) and runs an **FP32** matmul.
  FP32 matmul doesn't use BF16 tensor cores, but at m ≤ 8 that's irrelevant: it's memory-bound on the 8 MiB weight.
- **router, m > 8** → `mm(x, hi) + mm(x, lo)`, each a BF16 tensor-core GEMM accumulating into FP32, then an add.
  That's **three kernels** (two GEMMs + an elementwise add), and the FP32 output is written and re-read.

Not timed: the unpooler's all-gather across GPUs.

### Where a custom kernel might beat it (hypotheses to test, not facts)

1. **Small m (GEMV / skinny GEMM):** cuBLAS's general tiles are designed for big GEMMs. At m = 1–32 a dedicated
   kernel that just streams W at max bandwidth may get closer to the floor. Shapes with small N (qkv_gate N = 2304,
   router N = 256, sliced unpooler N = 1024) give few output tiles for the H200's 132 SMs (streaming
   multiprocessors, the GPU's "cores"), so work must be split along K ("split-K") to keep all SMs busy.
2. **Router at m > 8:** fuse the two GEMMs and the add into one kernel that reads hi and lo together.
3. **Router at m ≤ 8:** the cast + FP32 matmul could be one kernel reading the FP32 weight directly.
4. **m = 128 / 256:** hardest; cuBLAS is strong here. Likely the last thing to attack.

## 6. Correctness rules that constrain tricks

- Relative L2 error ≤ 1% (BF16 layers), ≤ **0.01%** for the router; any single row ≤ 5× that.
- **Probe rows:** ~1 in 8 rows of x has one nonzero entry (a power of two), so the output row is exactly a scaled
  column of W. Must be within 0.1%. → You **cannot** quantize weights to FP8/FP4; BF16 weights must stay BF16.
- **Router:** plain BF16 weights fail the 0.01% bound; FP32 or the hi + lo split both pass.
- Accumulate in FP32 (standard for tensor-core matmuls); output in BF16 (router FP32).
- Must also pass odd test shapes: m = 5, 7, 9, 33, 130, 512, 1000, and "outlier" inputs (1% of channels ×30).
  So a kernel must handle m that isn't a multiple of 8/16, and the unpooler switch at m ≤ 8 (N = 1024) vs m > 8 (N = 8192).
- Runs under **CUDA graphs**: no `.item()`, no syncs, no allocation or branching on input *values*
  (branching on m / layer is fine).

## 7. The starter `submission.py`

Plain PyTorch `x @ w.T` (router: `x.float() @ w.T` in FP32). Correct, and it calls cuBLAS under the hood, so it
should be roughly SGLang-speed on most shapes (but slower on the router at m > 8, where it does a full FP32
matmul instead of the hi/lo BF16 pair). It's our v000 baseline to measure against.

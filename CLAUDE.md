# CLAUDE.md — Open Kernel Challenge study project

## Who I am and what this repo is
I'm a student learning GPU kernel engineering. This repo is a public study log for the
Muse Spark Open Kernel Challenge, problem `muse_bf16_proj` (dense BF16 projections on 1×H200,
timed against SGLang). The goal is LEARNING and a CORRECT submission, not winning.

## Always read first
- `muse_bf16_proj/TASK.md` and `muse_bf16_proj/AGENTS.md` (official rules; they override this file)
- `notes/learning-log.md` (what has already been tried)

## How to work: one experiment per loop
1. **Explain** the idea in plain language before writing code: what it changes, and WHY it
   should be faster (memory-bound vs compute-bound, launches, tiling, etc.).
2. **Implement** it in a new file `muse_bf16_proj/variants/vNNN_<short-name>.py`.
   Never overwrite older variants. `submission.py` only ever holds the best PASSING version.
3. **Ask me before running `okc check`.** Checks use a limited daily quota.
   Tell me which file and what result you expect.
4. **Record** the result in `notes/learning-log.md` (template below), even if it got slower or failed.
5. **Commit** with a message like `v003: <idea> — <speedup or FAIL>`.

## Hard rules
- NEVER run `okc submit` without my explicit "yes, submit".
- Correctness beats speed. A failing kernel is worthless, however fast.
- Change one thing at a time so results are attributable.
- Don't read or print anything from `~/.okc` (auth token lives there). Never commit secrets.
- If unsure what the rules allow (e.g., calling library kernels), say so and check AGENTS.md / TASK.md.

## Teaching style
- I'm a beginner: define terms the first time you use them, use small concrete numbers
  (e.g., "this matrix is 1×7168 times 7168×512, so we read ~7 MB of weights for very little math").
- After each experiment, give me 2–3 sentences on the concept it illustrates.
- When a concept is worth keeping, add or extend a page in `notes/concepts/` (e.g., `gemv-vs-gemm.md`).

## learning-log.md entry template
```
### vNNN — <idea>
- Date:
- Hypothesis: why this should be faster
- Change: what the code does differently
- Result: PASS/FAIL, speedup (per-shape notes if available)
- What I learned:
- Next idea:
```
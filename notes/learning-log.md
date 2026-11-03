# Learning log — `muse_bf16_proj`

One entry per experiment, newest at the bottom. Record failures and slowdowns too.
Background: [concepts/problem-overview.md](concepts/problem-overview.md).

Template:
```
### vNNN — <idea>
- Date:
- Hypothesis: why this should be faster
- Change: what the code does differently
- Result: PASS/FAIL, speedup (per-shape notes if available)
- What I learned:
- Next idea:
```

---

### v000 — starter (plain PyTorch, unchanged `submission.py`)
- Date: 2026-11-02
- Hypothesis: none. This is the reference point. It calls cuBLAS via `x @ w.T`, so it should be close to SGLang
  except on the router at m > 8 (full FP32 matmul vs SGLang's BF16 hi + lo pair).
- Change: none.
- Result: not checked yet.
- What I learned: orientation only. All 30 scored shapes are memory-bound or (at m = 256) near the ridge;
  pooler + unpooler are 54–68% of each load's floor time. See problem-overview.md.
- Next idea: run `okc check` on the starter to get SGLang's real per-shape times and see which shapes are furthest
  from their floor.

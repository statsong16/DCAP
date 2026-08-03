# CLAUDE.md

Working agreement for this repository.

Sections 1–4 adapt the four principles from
[multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)
(Karpathy's observations on LLM coding failure modes).
Sections 5–9 are research-specific and take precedence when they conflict.

**Tradeoff:** these rules favour caution over speed. Use judgement on trivial tasks.

---

## 1. Think before coding

State assumptions out loud instead of quietly picking one.

- If a request has more than one reasonable reading, list them and ask.
- If a simpler approach exists, say so before implementing the complex one.
- If something is unclear, stop and name the confusion. Do not guess and continue.
- Surface tradeoffs rather than hiding them behind a choice already made.

## 2. Simplicity first

Write the minimum that solves the stated problem.

- No features that were not requested.
- No abstraction layers for code used once.
- No configurability "for later".
- If 200 lines could be 50, rewrite it as 50.

## 3. Surgical changes

Every changed line should trace back to the request.

- Do not reformat, rename, or "improve" adjacent code.
- Match the existing style even where you would write it differently.
- Remove only the imports and helpers that *your* change orphaned.
- If you spot unrelated dead code, mention it — do not delete it.

## 4. Goal-driven execution

Turn tasks into verifiable goals before starting.

- "Add the phylogeny mask" → "assert the attention bias changes encoder output, then report sparsity"
- "Fix the leakage" → "write a check that no study appears in both folds, then make it pass"

For multi-step work, state the plan first:

```
1. [step] → verify: [check]
2. [step] → verify: [check]
```

---

## 5. Never report a number you did not compute

This is the one rule with no exceptions.

- Every metric in a README, notebook, or figure must come from code in this repo
  that was actually executed.
- If a run has not happened yet, write `TBD` — never a plausible placeholder.
- Do not "estimate" an AUC, a p-value, a sample size, or a runtime.
- If asked to summarise results, read the output files. Do not recall them.

## 6. No data leakage

Cohort data is pooled across studies, sites, and visits. Leakage is the default
failure mode, not an edge case.

- Split by **group**, not by row: study, site, or subject — never sample.
- Repeated measures from one subject never straddle a train/test boundary.
- Preprocessing that learns parameters (scaling, bin edges, imputation, feature
  selection) is fit on training folds only and applied to held-out folds.
- Any new modelling script must state its splitting strategy in a comment at the top.

## 7. No participant data in the repository

- `data/` is git-ignored. Nothing derived from real participants is committed —
  not counts, not IDs, not subsetted tables.
- Every pipeline must run end-to-end on synthetic data so the repo is
  reproducible without a data access agreement.
- Figures committed to the repo must not allow re-identification.

## 8. Reproducibility is part of "done"

- Set and record random seeds. Report the seed alongside any result.
- Pin versions in `requirements.txt` / `renv.lock`.
- One documented command reproduces the reported table from raw inputs.
- Long-running steps write intermediate artefacts; they do not live only in a notebook.

## 9. Report negative results

- A model that loses to the baseline stays in the results table.
- Baselines are run every time, not once at the beginning.
- Report calibration alongside discrimination. AUC alone is not a result.
- If a finding weakened after a fix, say so in the commit message.

---

**These rules are working if:** diffs contain only requested changes, results
tables are reproducible from a clean clone, and every number in the README can be
traced to a script and a seed.

# model-governance-evidence-kit

**The eval harness scores agent outputs per case; this produces periodic model-level validation evidence for an auditor.**

A periodic validation pack for a credit model, in the style of `collections-risk-scorer` (which builds a score) and `credit-retrospective-kit` (which audits one book once): this one answers the recurring governance question — *is the model still behaving the way it did when it was approved, and can we prove it with a reproducible artifact?* It compares a `current` scored window against the approved `reference` window on exactly three checks, and emits a pass/fail evidence pack (Markdown + JSON) with SHA-256 hashes of the inputs and fixed run metadata, so an auditor can re-run it and get byte-identical output.

**Not to be confused with `agent-slo-tracker`:** that repo operates live agent SLOs — service-level objectives, watched continuously in production. This repo is periodic and offline: one validation snapshot per review period, for a model-governance file, not a live operations dashboard.

> **Synthetic data only.** Every record is fabricated by a seeded generator. No real borrower, account, or portfolio data is used anywhere.

Stdlib-only Python. Zero network, zero API keys, zero external dependencies.

## Quickstart

```bash
# 1. Generate the seeded portfolio (800 reference + 800 current, seed 42)
python3 cli.py generate --out portfolio.jsonl

# 2. Run the three validation checks (JSON)
python3 cli.py validate --input portfolio.jsonl

# 3. Build the auditor evidence pack (JSON, or --format md)
python3 cli.py evidence --input portfolio.jsonl --format md

# 4. CI gate: exit 0 = pass/warn, 1 = validation fail, 2 = data/IO error
python3 cli.py check --input portfolio.jsonl; echo $?

# 5. One-command end-to-end demo
python3 demo.py

# 6. Tiny localhost API: POST /validate, GET /health on 127.0.0.1
python3 cli.py serve --port 8765
curl -s -X POST localhost:8765/validate -d '{}'
```

Every command except `generate` also works with no `--input`, using the seeded portfolio.

## The three checks (and only these)

1. **Drift — PSI/CSI vs the reference window.** **PSI** — population stability index, a standard drift statistic: `PSI = Σ (cur% − ref%) × ln(cur% / ref%)` over binned distributions, each share floored at epsilon 0.0001 so an empty bin cannot divide by zero. Applied to the score it is PSI; applied to each input feature it is **CSI** — characteristic stability index, the same formula per feature. Verdicts: < 0.10 pass, 0.10–0.25 warn, > 0.25 fail.
2. **Run-to-run variance.** The same model re-scored over the same cases under 5 seeds (jitter sd 0.01). Mean per-case score std ≤ 0.02 passes, > 0.04 fails — the score an auditor sees should be stable, not a lucky draw.
3. **Accuracy vs labels (current window).** **AUC** — area under the ROC curve; the probability a randomly chosen defaulter was scored riskier than a non-defaulter — computed from scratch with the Mann–Whitney rank formula (ties take average ranks) and cross-checked against the brute-force pairwise definition. Plus **calibration** — in each score band, mean predicted score vs the share that actually defaulted; max band error > 0.20 fails, > 0.10 warns.

## Headline verdicts on the seeded portfolio (seed 42, 1,600 records)

The generator plants two defects in the `current` window — a drift shift in `dti_ratio` (+0.18 mean, scores +0.06 with it) and a miscalibrated S3 band (scores 0.30–0.45 defaulting at ~85% instead of ~37%) — and the pack finds both:

| Check | Result | Verdict |
|---|---|---|
| Score PSI | 2.612705 | **fail** |
| `dti_ratio` CSI | 2.924964 | **fail** |
| `income_volatility` CSI / `credit_utilization` CSI | 0.016549 / 0.008553 | pass |
| AUC (current) | 0.777225 | pass |
| Calibration — worst band S3: predicted 0.367656 vs observed 0.862416 | max error 0.49476 | **fail** |
| Run-to-run variance (5 runs) | mean std 0.007989 | pass |
| **Overall evidence verdict** | | **FAIL** |

The undrifted, non-miscalibrated control (`--no-drift --no-miscalibration`) passes everything: score PSI 0.008779, max calibration error 0.095941, overall **PASS**, `check` exit 0 — against exit 1 on the seeded portfolio and exit 2 on a missing/corrupt file. The undrifted features stay clean in the seeded pack too, which is the point: the drift check names *which* distribution moved, not just that something did.

## Evidence pack determinism

The pack carries the model name (`synthetic-credit-score-v1`), a timestamp fixed for determinism (`2026-11-04T00:00:00Z`), the thresholds used, a verdict per check, and the SHA-256 of the canonical input records. It is a pure function of its inputs: the JSON pack's md5 is `564d8b547217c9e1ad6e6d203b04efe0`, byte-identical across verified runs (pinned by eval G5).

## Architecture

| Module | Job |
|---|---|
| `generator.py` | Deterministic synthetic scored portfolio: features, score, label; reference + current windows; the planted drift shift and miscalibrated band (both switchable off for a clean control) |
| `drift.py` | PSI (score) and CSI (per feature) vs the reference window, epsilon-floored shares, pass/warn/fail verdicts |
| `metrics.py` | AUC from scratch (Mann–Whitney, tie-averaged) + brute-force pairwise cross-check, calibration by score band, seeded re-run variance/stability |
| `evidence.py` | Evidence pack assembly: fixed metadata, SHA-256 input hash, per-check verdicts, overall verdict, Markdown rendering — a pure function of the records |
| `validation.py` | Data-quality gate: required fields, ranges, window vocabulary, duplicate ids, bool traps on the label |
| `cli.py` | `generate` / `validate` / `evidence` / `check` (CI exit 0/1/2) / `demo` / `serve` (stdlib `http.server` on 127.0.0.1: `POST /validate` with `{"records": [...]}` or an empty body for the seeded portfolio, `GET /health`) |
| `demo.py` | End-to-end narrative with asserts: determinism, planted-vs-control drift, AUC vs brute force, bad band, variance, evidence determinism, check exit codes, live serve round-trip |
| `fixtures/` | 40-record sample portfolio (`sample_portfolio.jsonl` + `.csv`, seed 42, 20 per window) |

## Evals

`python3 evals/run_evals.py` → **10/10 golden evals**, report committed at `evals/eval_report.json` (md5 `42b15ecba8b5f841fb37e8ed2a462c20`, byte-identical across 3 verified runs):

- **G1** PSI flags the planted drift (score PSI 2.612705, `dti_ratio` CSI 2.924964, verdict fail)
- **G2** PSI clean on the undrifted control (0.008779, pass)
- **G3** calibration catches the planted bad band (S3 error 0.49476, worst band S3, fail)
- **G4** AUC matches brute-force pairwise on a small fixture, plus a hand-computed 0.75 case
- **G5** evidence pack determinism — byte-identical JSON across runs, input hash and verdicts pinned
- **G6** generator determinism (JSONL md5 pinned) + headline AUC 0.777225 pinned
- **G7** variance stable on seeded runs (mean std 0.007989, pass) and failing under large jitter
- **G8** CLI `check` exit codes 1 (planted) / 0 (control) / 2 (missing file)
- **G9** serve round-trip matches the local evidence pack exactly
- **G10** validation flags a planted defect; control pack's overall verdict is pass

Unit tests: `python3 -m unittest discover -s tests` → **58 hermetic tests**, including a no-network enforcement test (sockets stubbed), hand-computed PSI and AUC cases (perfect / reversed / tied), the Mann–Whitney AUC verified against brute-force pairwise on random tied fixtures and a generated slice, planted-drift and planted-miscalibration known-answer tests, epsilon handling for empty bins, variance stability and instability, evidence byte-determinism, CLI exit codes, and serve round-trips.

## Development loop — what prototyping actually caught

**Calibration on tiny bands cries wolf.** In the first prototype the seeded S5 band (score ≥ 0.60) held only **5 current-window cases** and read a 0.415 calibration error — larger than many real defects would produce — purely from small-n noise, while the genuinely planted S3 band held 298 cases. The pack as shipped still reports every band honestly (the planted band remains the worst, at 0.49476), but the find stands as the documented limit below: a production version of this pack needs a minimum-n rule per band (suppress or widen bands under ~30 cases) before a human should act on a single band's error. Raising the generator's base propensity to fatten the top bands was tried and rejected — it dragged the control portfolio's AUC and calibration into warn territory and made the clean case less clean, which is the worse trade for a governance demo.

## Scope and honest limits

- **Three checks only, by design.** No data-lineage map, no segment slicing, no fairness testing, no live monitoring — a periodic pack, not a platform. (Lineage was considered and cut in review.)
- **The planted defects are planted.** Drift and the bad band are built into the generator at known sizes, so "the checks find them" proves the measurement machinery, not that these effect sizes resemble a real portfolio's decay.
- **Calibration bands have no minimum-n rule** (see the dev-loop find): sparse bands are noisy, and the seeded S5 band shows it.
- **Variance is simulated re-runs, not a real model's nondeterminism.** The re-score jitter (sd 0.01) stands in for retraining/inference variance; a production pack would re-run the actual model artifact.
- **The score is a noisy propensity, not a fitted model.** Its AUC of 0.777 is a property of the planted noise and the planted bad band sitting outside the current window's ranking path.
- **The API is a demo surface.** It binds 127.0.0.1 only, holds its portfolio in memory, and resets on restart.

## Why this project

Model governance is where forward-deployed and data-science work meets accountability: a model that was fine at approval can drift, miscalibrate, or go unstable, and someone has to produce evidence — not vibes — on a schedule. This repo builds that evidence the way an auditor would want it: defined formulas, stated thresholds, planted defects the checks must find, a clean control they must pass, and a pack that hashes its inputs and reproduces byte-for-byte.

## License

MIT — see LICENSE.

# Project Setup & Progress

- [x] Clarify requirements and scope (NCAAB betting engine w/ Snapdragon NPU inference via ONNX Runtime DirectML/QNN when available)
- [x] Scaffold Python project structure with `src` package, CLI, and modules
- [x] Install dependencies and configure Python environment
- [x] Run synthetic pipeline to validate setup (baseline predictions + odds join + Flask render)
- [x] Fill real data adapters (APIs/feeds) and iterate on models
- [x] Provider event probing CLI added (`probe-odds-events`) to validate coverage vs games
- [x] Finalize workflow implemented (`finalize-day` CLI + `/api/finalize-day` + UI button) with half-time derivation support
- [x] Results surfacing endpoints/pages: stake sheet (orig/cal/compare + CSV download), coverage diagnostics, calibration artifact, finals aggregation
- [x] Added `/api/results` structured endpoint for per-date JSON consumption
- [x] Daily PowerShell script now generates stake sheets (baseline + calibrated) and comparison; optional previous-day finalize wired in

Next Targets:
- [ ] Expand synthetic pipeline to exercise segmented + calibration + stake sizing end-to-end automatically
- [ ] Add historical archive navigation for picks & results (multi-date browsing)
- [ ] Introduce asynchronous polling for live games and finalize auto-trigger hints

Notes:
- ONNX Runtime wheel still unavailable via pip for Windows ARM64 here; prediction commands gracefully prompt for manual ORT install (CPU/DirectML) or a custom build with QNN EP.
- TheOddsAPI key support via `.env` and `fetch-odds` CLI command working; odds join populates `games_with_last.csv`.
- Preseason blending available in `daily-run` via `--preseason-weight` and `--preseason-only-sparse`; train priors-only models first with `train-preseason` to enable blend.
- Coverage diagnostics: use `probe-odds-events` to enumerate provider-listed events; intersection counts confirm partial provider scope on sparse/exhibition-heavy slates.
- Finalization now writes `daily_results/results_<date>.csv` with ATS/OU outcomes and optional half metrics.
- Daily automation: `scripts/daily_update.ps1` covers prior-day reconciliation + finalize, today's run, and stake sheet generation + CSV compare. Flags: `-SkipFinalizePrev`, `-SkipStakeSheets`.

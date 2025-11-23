# NCAAB Predictive Betting Engine (Snapdragon NPU/DirectML)

This project is a college basketball (NCAAB) predictive engine focused on:

- Game totals (full game)
- 1H/2H totals
- Game winner
- Against the spread (ATS)

It is designed to run fast inference on Windows Copilot+/Snapdragon systems via ONNX Runtime with the DirectML execution provider (leveraging the NPU when supported). If available, it can also prefer the Qualcomm QNN Execution Provider for direct NPU acceleration.

## Quick start

1. Create a Python 3.10+ venv (Windows PowerShell):

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. If ONNX Runtime wheels are unavailable for your Windows ARM64 environment, use a local ORT build by setting a DLL directory or run the helper script:

```powershell
# Point to the folder that contains onnxruntime DLLs and provider DLLs
$env:NCAAB_ORT_DLL_DIR = "C:\\path\\to\\onnxruntime-qnn-build\\bin"

# Optional: QNN SDK root and/or explicit backend DLL
$env:NCAAB_QNN_SDK_DIR = "C:\\Qualcomm\\QNN_SDK"
# $env:NCAAB_QNN_BACKEND_DLL = "C:\\Qualcomm\\QNN_SDK\\lib\\htp\\Windows\\Release\\qnn-htp.dll"

# If you have a local onnxruntime wheel or unpacked module directory, point Python at it:
# (semicolon separate multiple entries; you can pass a .whl file path or a directory)
$env:NCAAB_ORT_PY_DIR = "C:\\path\\to\\onnxruntime-<ver>-cp311-cp311-win_arm64.whl"

# Or use the helper script to set everything and verify providers:
# (edit the paths for your machine)
pwsh -File scripts/enable_ort_qnn.ps1 -OrtBinDir "C:\\path\\to\\onnxruntime-qnn-build\\bin" -OrtPyWheelOrDir "C:\\path\\to\\onnxruntime-<ver>-cp311-cp311-win_arm64.whl" -QnnSdkDir "C:\\Qualcomm\\QNN_SDK" -InstallWheel
```

4. Run a tiny ONNX smoke test (creates a dummy ONNX model and runs inference via QNN if available, else DML, else CPU):

```powershell
python -m scripts.synthetic_demo --make-model --predict
```

Expected: it reports available ONNX providers, tries QNNExecutionProvider (if configured), else DmlExecutionProvider, and prints a small prediction batch.

You can also inspect providers with:

```powershell
python -m ncaab_model.cli ort-info
```

## Season-start backfill (last two seasons)

To initialize the model with enough history before the season tips off, run the end-to-end backfill, training, and evaluation for the last two seasons:

```powershell
python -m ncaab_model.cli evaluate-last2 --closing none
```

This will:

- Fetch 2023–24 and 2024–25 games into `outputs/games_*.csv`,
- Fetch ESPN boxscores and compute four-factors into `outputs/boxscores_last2.csv`,
- Build enriched features (schedule, adjusted ratings, four-factors) into `outputs/features_last2.csv`,
- Train baseline models and export ONNX into `outputs/models/`,
- Score predictions into `outputs/predictions_last2.csv`,
- Write accuracy reports into `outputs/eval_last2/`.

Optional: If you have access to TheOddsAPI historical endpoint, you can add closing-line evaluation by setting `--closing history` (or use `current` to snapshot available books per day):

```powershell
python -m ncaab_model.cli evaluate-last2 --closing history
```

Reports are written to `outputs/eval_last2/` and include per-game and summary CSV/JSON files.

## Project layout

- `src/ncaab_model` — library code
  - `onnx/export.py` — build/export ONNX models (includes a tiny dummy model generator)
  - `onnx/infer.py` — ONNX Runtime inference with QNN/DirectML preference, CPU fallback, and Windows DLL bootstrap via `NCAAB_ORT_DLL_DIR`
  - `cli.py` — Typer CLI entry points
  - `config.py` — settings (paths, API keys)
  - Future modules: `data/`, `features/`, `models/`, `train/`, `eval/`
- `scripts/` — runnable helpers (synthetic demo)
- `tests/` — minimal test(s) to validate setup
- `outputs/` — artifacts (models, reports)

## NPU acceleration (DirectML and Qualcomm QNN)

We use ONNX Runtime with the `DmlExecutionProvider` on Windows to target supported GPUs and NPUs (e.g., Snapdragon X Elite). If DML is not available, we fall back to CPU automatically.

If you have the Qualcomm QNN SDK installed (e.g., at `C:\\Qualcomm\\QNN_SDK`) and an ONNX Runtime build that includes the QNN Execution Provider, the engine will preferentially use `QNNExecutionProvider`.

Environment variables and settings:

- `NCAAB_QNN_SDK_DIR` — root of your QNN SDK (defaults to `C:\\Qualcomm\\QNN_SDK` if present)
- `NCAAB_QNN_BACKEND_DLL` — optional absolute path to the QNN backend DLL to use
 - `NCAAB_ORT_DLL_DIR` — optional directory containing ONNX Runtime DLLs (use this when pip wheels are not available)

Note: An onnxruntime wheel with QNN EP may not be published for all Python/arch combos. If unavailable, use DirectML (`onnxruntime-directml`) or CPU (`onnxruntime`) until a compatible QNN EP build is installed. The code automatically falls back.

### Runtime provider inspection & benchmarking

The Flask app exposes two lightweight diagnostics endpoints once it is running (`python app.py`):

- `GET /api/ort-providers` — returns JSON with:
  - `available_providers`: providers reported by `onnxruntime.get_available_providers()` (e.g. `["DmlExecutionProvider","CPUExecutionProvider"]`)
  - `session_providers`: the actual ordered providers used in a sample test model session (QNN → DML → CPU preference)
  - `dll_dir`: value of `NCAAB_ORT_DLL_DIR` if set
  - `qnn_sdk_root`: QNN SDK root if detected
- `GET /api/ort-benchmark` — runs a quick synthetic inference benchmark (32 warmups + 64 timed runs) against the first available small test model (`mlp_megatron_basic_test.onnx`, `bart_mlp_megatron_basic_test.onnx`, or `self_attention_megatron_basic_test.onnx`). Returns:
  - `avg_ms`: average per-inference latency in milliseconds
  - `session_providers`: provider order actually used
  - `available_providers`: same as above
  - `model`: test model file name

If no test model is present, `/api/ort-benchmark` responds with `404` and an explanatory message. Add one of the included ONNX test artifacts (already in repo root) to enable benchmarking.

Benchmark endpoint caching: Results are cached for 120 seconds to avoid repeated warmups. The JSON payload includes `"cached": true` when served from cache.

Health endpoint:

`GET /api/health` returns a small JSON document:

```json
{
  "status": "ok",
  "providers": ["DmlExecutionProvider","CPUExecutionProvider"],
  "last_pipeline_stats": {"pred_total_uniform_flag": false, ...},
  "timestamp": "2025-11-22T15:34:12.123456Z"
}
```

`last_pipeline_stats` mirrors the most recent in-request diagnostic frame generated during prediction assembly (e.g., uniform total flags, coverage counts). If no request has populated stats yet it will be `null`.

Provider priority logic is encapsulated in `src/ncaab_model/onnx/infer.py`:

1. If QNN EP is available and valid options can be constructed, append `QNNExecutionProvider`.
2. If DirectML EP is available append `DmlExecutionProvider`.
3. Always append `CPUExecutionProvider` as a safety fallback.

This ensures prediction paths prefer accelerated execution without manual flagging. If you build ONNX Runtime locally with only DirectML, QNN will simply be omitted.

To confirm acceleration via CLI instead of HTTP:

```powershell
python -m ncaab_model.cli ort-info
```

Or run the verification helper:

```powershell
python scripts/ort_verify.py
```

For a quick manual benchmark using the HTTP endpoint:

```powershell
Invoke-RestMethod http://localhost:5050/api/ort-benchmark | ConvertTo-Json
```

Expect lower `avg_ms` after enabling DirectML (and potentially even lower with QNN on supported hardware/models).

## Remote Predictions Parity & Ingestion

Ensure the hosted Flask app displays identical prediction values to your local artifacts and never falls back to synthetic shells.

### Promotion vs Shell Logic

`_load_predictions_current()` executes deterministic steps:

1. Use `predictions_<date>.csv` if present.
2. If missing / shell (both `pred_total` & `pred_margin` absent or all NaN) / stale vs model artifact → promote from:
   - `predictions_model_calibrated_<date>.csv` (preferred)
   - else `predictions_model_<date>.csv`
3. If no model artifact exists: synthesize a shell from `games_curr.csv` (metadata only) so the UI can render a slate.

Shell creation only occurs when no real model outputs exist; pushing a real file prevents shells entirely.

### Ingestion Endpoints

| Endpoint | Method | Body | Writes |
|----------|--------|------|--------|
| `/api/ingest/predictions` | POST | CSV (multipart `file` or raw `text/csv`) | `predictions_<date>.csv` |
| `/api/ingest/model-predictions` | POST | CSV (multipart `file`) | `predictions_model[_calibrated]_<date>.csv` |

Auth (optional): set `NCAAB_INGEST_TOKEN` and send header `X-Ingest-Token: <token>`. If unset, endpoints are open.

Response JSON includes `md5` hash for parity verification.

### Slim Push Script

`scripts/push_slim_predictions.py` picks the best local source (calibrated → raw → existing promoted) and uploads only:

```
game_id,date,pred_total,pred_margin
```

Example:

```powershell
python scripts/push_slim_predictions.py --date 2025-11-22 --url https://your-render-app.onrender.com/api/ingest/predictions --token $Env:NCAAB_INGEST_TOKEN
```

Script output prints local & remote md5 hashes; equality confirms parity.

### Manual PowerShell Uploads

```powershell
$file = "outputs/predictions_2025-11-22.csv"
Invoke-RestMethod -Uri "https://your-render-app.onrender.com/api/ingest/predictions" -Method Post -InFile $file -ContentType 'text/csv' -Headers @{ 'X-Ingest-Token' = $Env:NCAAB_INGEST_TOKEN }

$model = "outputs/predictions_model_calibrated_2025-11-22.csv"
Invoke-RestMethod -Uri "https://your-render-app.onrender.com/api/ingest/model-predictions" -Method Post -InFile $model -ContentType 'text/csv' -Headers @{ 'X-Ingest-Token' = $Env:NCAAB_INGEST_TOKEN }
```

### Environment Variables

| Var | Purpose |
|-----|---------|
| `NCAAB_INGEST_TOKEN` | Shared secret required in `X-Ingest-Token` header |
| `NCAAB_PREDICTIONS_FILE` | Override predictions load path |
| `NCAAB_MODEL_PREDICTIONS_FILE` | Override model artifact path |
| `NCAAB_OUTPUTS_DIR` | Force outputs directory resolution |

### Daily Parity Flow

1. Run `daily-run` + any calibration.
2. Push slim predictions.
3. Optionally push model predictions artifact.
4. Visit `/?diag=1` and check `preds_load_rows` & source path.
5. Confirm md5 matches local file.

### Optional GitHub Action

```yaml
name: Push Slim Predictions
on:
  schedule:
    - cron: '15 15 * * *'
  workflow_dispatch:
jobs:
  parity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - name: Daily run (light)
        run: python -m ncaab_model.cli daily-run --threshold 2.0 --target-picks 10
      - name: Push slim predictions
        env:
          NCAAB_INGEST_TOKEN: ${{ secrets.NCAAB_INGEST_TOKEN }}
        run: python scripts/push_slim_predictions.py --date $(date +%F) --url https://your-render-app.onrender.com/api/ingest/predictions --token $NCAAB_INGEST_TOKEN
```

### Security Notes

- Use a long random `NCAAB_INGEST_TOKEN`; rotate regularly.
- Endpoints validate structure + hash only; add stricter checks if public.
- Legacy verbose diagnostics endpoint removed—use `/?diag=1` and `/api/health` (`last_pipeline_stats`).

---

Remote prediction parity is now deterministic and independent of startup timing.

## CLI commands (highlights)

- Data
  - `fetch-games` — fetch real games (ESPN/NCAA) to Parquet/CSV
  - `fetch-boxscores` — ESPN box scores + possessions/four-factors
  - `fetch-odds` / `fetch-odds-history` — current or historical odds snapshots
  - `make-closing-lines` — aggregate snapshots into per-book closing lines
  - `make-last-odds` — select the strict last observed pre-tip odds per (event_id, book, market, period) with optional timestamp tolerance (no synthetic fallbacks)
  - `join-odds` / `join-closing` — join odds/closing to games
  - `join-last-odds` — join last pre-tip odds to games (per book, preserves multi-row markets)
  - `seed-team-map` — build/maintain a canonical team name map
- Features & modeling
  - `build-features` — rolling team features + schedule + opponent-adjusted ratings + four-factors
  - `train-baseline` — ridge baselines (totals/margin) and ONNX export
  - `predict-baseline` — score features using ONNX (QNN/DML/CPU) or NumPy fallback
- Evaluation & operations
  - `eval-accuracy` — MAE/RMSE/bias/R² for totals and margins
  - `eval-accuracy-closing` — model vs closing (MAE delta, beat rate, edge correlation)
  - `backtest` / `backtest-closing` — totals strategy PnL using current/closing lines
  - `make-picks` — generate a clean pick sheet
  - `evaluate-last2` — one-shot seasonal backfill + train + eval for last two seasons
  - `daily-run` — fetch odds & games for today, build features, predict, generate picks, and ingest to SQLite
  - `produce-picks` — generate expanded multi-market picks (totals/spreads/moneyline; full game + halves) into `outputs/picks_raw.csv`
  - `daily-results` / `daily-results-range` — reconcile per-day outcomes; now supports `--picks-raw-path` to grade expanded picks and writes `outputs/daily_results/picks_raw_results_YYYY-MM-DD.csv`
  - `update-branding` — fetch Division I team logos/colors (ESPN) into `data/team_branding.csv`
  - `ingest-outputs` — bulk ingest common CSV artifacts into SQLite with automatic schema evolution
  - `rebuild-db` (optional) — drop specified tables or entire database (if added) for a clean ingest

## Daily run

For in-season operation, use the daily pipeline:

```powershell
python -m ncaab_model.cli daily-run --threshold 2.0 --default-price -110 --book-whitelist "fanduel,betmgm" --target-picks 10
```

This fetches today’s games and odds, builds features, predicts, selects picks, and ingests outputs into SQLite (`data/ncaab.sqlite`).

## Backtesting, calibration, and staking

- Generate per-day backtest metrics (uses book-level closing when present; falls back to per-game medians):

```powershell
python scripts/daily_backtest.py --date 2025-11-19
```

Outputs `outputs/backtest_metrics_<date>.json` including summaries for totals/spread/moneyline and a `bets_detail` section.

- Backfill core artifacts across a range and refresh season rollups + spread logistic calibration:

```powershell
python scripts/backfill_artifacts.py --start 2025-11-01 --end 2025-11-20
# optional flags: --skip-calibration --skip-season
```

This writes daily residuals/scoring/reliability/backtest artifacts, then runs `scripts/calibrate_spread_logistic.py` and `scripts/season_aggregate.py`.

Provisional calibration: when recent data is sparse, the calibrator now emits a provisional `calibration_spread_logistic.json` once it has at least 25–50 graded spread rows. The payload includes `provisional: true` and will be refined automatically as more days accrue. If there are no usable rows, it carries forward the prior K (if any) or falls back to a conservative default `K=0.115`.

- Simulate alternative staking regimes on stored picks for a date:

```powershell
python scripts/stake_simulation.py --date 2025-11-19 --mode kelly --kelly_fractions 0.25 0.5 1.0
python scripts/stake_simulation.py --date 2025-11-19 --mode flat --flat_units 0.5 1 2
```

The Flask app now also computes per-game spread Kelly suggestions using a calibrated logistic constant `K` from `outputs/calibration_spread_logistic.json` (if present); see `kelly_frac_spread` and `kelly_side_spread` columns in the server dataframe.
On the main Cards page, a small Kelly badge appears in the Full Game → Odds row. It shows the suggested Kelly fraction at -110 and direction (HOME/AWAY) based on `(pred_margin - spread)`.

## Team Branding (Logos & Colors)

Run once (and periodically) to pull current Division I team branding from ESPN:

```powershell
python -m ncaab_model.cli update-branding
```

This writes `data/team_branding.csv` with columns:

- `team` — display name
- `logo` — HTTPS logo asset (prefer "full" or "default" rendition)
- `primary_color`, `secondary_color` — hex colors normalized to `#RRGGBB`
- `text_color` — auto-generated readable text color for badges
- `espn_id`, `abbreviation` — identifiers for cross-referencing

The Flask app (`app.py`) enriches prediction rows with branding data. If a logo or color is missing, the UI falls back gracefully (basketball emoji + neutral pill color).

Trademark note: Team names, logos, and color schemes may be trademarked. Assets are fetched from ESPN’s public API. Use them for personal/analysis purposes; do not redistribute commercially without appropriate rights.

Regenerate as teams update branding mid-season:

```powershell
python -m ncaab_model.cli update-branding --overwrite True
```

## SQLite Ingestion & Schema Evolution

Outputs can be ingested into SQLite for downstream queries/dashboards:

```powershell
python -m ncaab_model.cli ingest-outputs
```

Tables created (with upsert keys):

- `games` (game_id)
- `boxscores` (game_id)
- `odds_current` (dynamic composite; falls back when columns missing)
- `closing_lines` (event_id, book, market, period)
- `last_odds` (event_id, book, market, period) — optional if you generate and ingest strict last pre-tip odds
- `features` (game_id)
- `predictions` (game_id)
- `picks` (game_id, book)

Schema evolution: When new columns appear in CSVs (e.g., `home_pf15`, `model_used`), the ingest process adds them automatically via `ALTER TABLE` without dropping existing data.

To force a clean slate (dropping and recreating selected tables) a future `--rebuild` flag may be added. Until then, you can manually drop a table:

```powershell
python -m ncaab_model.cli init-db  # ensures DB exists
sqlite3 data\ncaab.sqlite "DROP TABLE IF EXISTS predictions;"
python -m ncaab_model.cli ingest-outputs
```

## Regenerating Branding & Picks End-to-End (Daily Script)

The PowerShell automation script `scripts/daily_update.ps1` orchestrates:

1. Previous day games + odds snapshots → closing lines
  - (Optional) Also generate strict last odds: `make-last-odds` for higher integrity vs heuristic closing.
2. Daily results reconciliation & tuning update
3. Optional model retrain
4. Today’s fetch + predictions + picks + ingest

Run it after activating your venv:

```powershell
. .\scripts\daily_update.ps1 -NoCache
```

Logs stored under `outputs\logs\` with timestamped transcript.

## Next steps

- Improve totals modeling with richer tempo and shooting adjustments, and market-informed features near tipoff.
- Integrate an alternate historical odds source if TheOddsAPI odds-history is unavailable; expand closing-line evaluation across seasons.
- Add a multitask neural model (PyTorch) with ONNX export for accelerated inference on Snapdragon.
- Add explicit `--rebuild` option to `ingest-outputs` for table recreation.
- Provide a dashboard (Streamlit or front-end) for interactive query of SQLite summaries.

## Odds & Closing Lines Backfill

Historical closing coverage is low by default unless odds snapshots have been fetched for each date. Use the helper script to backfill:

```powershell
./scripts/backfill_odds_and_closing.ps1 -Start 2025-11-01 -End 2025-11-08
```

This performs:
1. `fetch-odds-history` across the range (current mode snapshots).
2. `make-closing-lines` to build per-book closing.
3. `join-closing` with fused games.
4. `validate-closing-coverage` writing `outputs/closing_coverage.csv` (per-date stats).

Re-run for earlier months (February/March) to raise postseason coverage.

### Closing Lines vs Last Odds (Integrity)

We now distinguish two concepts:

- Closing Lines (Heuristic): Attempts to pick a "closing" per book/market using priority rules (prefer last_update ≤ commence_time; else fetched_at ≤ commence; else final snapshot). This can include a small number of post-tip or fallback rows if true pre-tip snapshots are missing.
- Last Odds (Strict): The final observed snapshot timestamp (last_update if present else fetched_at) that is ≤ commence_time + tolerance (default 60s). Rows after tip or without a known commence_time are discarded. No synthetic substitution. If no pre-tip snapshot exists for a book, that book simply has no last odds entry.

Use `make-last-odds` + `join-last-odds` when you need 100% provenance ("real, observed" quotes only). Retain `make-closing-lines` for legacy evaluations or when approximate closing is acceptable.

Recommended labeling in UI:
- "Last (pre-tip)" for strict last odds values.
- "Market Median (Derived)" only when you aggregate across books (clearly mark as derived, not a direct quote).

Coverage Strategy:
- Schedule frequent snapshots (e.g., every 2–3 minutes leading up to tip) to maximize strict last odds completeness.
- Backfill historical dates via `fetch-odds-history`; then run `make-last-odds` to create `last_odds.csv`.

## Cleaning Placeholder Daily Results

Remove future-date placeholder result files (all zero scores, no predictions) to declutter:

```powershell
./scripts/cleanup_placeholder_daily_results.ps1
```

## Odds API Endpoint

Flask now exposes `/api/odds` returning per-game aggregated full-game totals odds quotes:

```json
{
  "n": 69,
  "rows": [
    {
      "game_id": "401812261",
      "market_total": 152.5,
      "commence_time": "2025-11-08 17:00",
      "quotes": [
        {"book": "BookA", "total": 152.5, "price_over": -110, "price_under": -110}
      ]
    }
  ]
}
```

### Picks API

Two endpoints are available:

- `/api/recommendations` — returns the clean pick sheet (`picks_clean.csv`) as JSON.
- `/api/picks_raw` — returns expanded picks from `picks_raw.csv` with optional `?date=YYYY-MM-DD` filter and branding enrichment (logo/colors for home/away when available).

UI pages:

- `/picks-raw` — server-rendered page listing expanded picks with simple filters (date, market, period) and edge highlighting.

## Neutral Site Indicator

Cards display a "Neutral Site" pill when `neutral_site` is true for the game.

## UI: archive navigation and live auto-refresh

- The Cards page offers an Archive Dates list for quick browsing and a Prev/Next stepper when viewing a specific date.
- A small Auto-refresh checkbox lets you keep the page updated every 60s. If there are live games and auto-refresh is off, a dot appears in the page title as a subtle hint.

## Historical Backfill Orchestrator

Use the PowerShell helper to accumulate history, rebuild last/closing lines, generate artifacts, calibrate spread K, and refresh season rollups. It will also train segmented models if `outputs/features_hist.csv` is present.

```powershell
# Backfill a specific span
.\n+scripts\historical_backfill.ps1 -Start 2023-11-01 -End 2024-04-15

# Or backfill typical season windows (Nov 1 .. Apr 15 next year)
.\n+scripts\historical_backfill.ps1 -Seasons 2023,2024
```

Outputs affected:
- `outputs/odds_history/` snapshots and `outputs/closing_lines.csv`
- `outputs/last_odds.csv` and per-day `outputs/games_with_last.csv`
- Daily artifacts under `outputs/daily_results/` and scoring/reliability/backtest CSVs
- `outputs/calibration_spread_logistic.json` (provisional until sufficient rows)
- Season summaries under `outputs/season/`
- Segmented models to `outputs/models_segmented*/` when features are available

## Segmented Models (team/conference)

Train per-segment ridge models (totals and margins) with a single command. Models are saved as JSONL entries with learned weights and scaling per segment.

```powershell
python -m ncaab_model.cli train-segmented outputs/features_hist.csv --segment team --out-dir outputs/models_segmented --min-rows 25 --alpha 1.0
python -m ncaab_model.cli train-segmented outputs/features_hist.csv --segment conference --out-dir outputs/models_segmented_conf --min-rows 25 --alpha 1.0
```

Notes:
- Input must include historical features with `game_id`, segment keys (`team` or `conference`), and target columns.
- Low-sample segments are skipped (`--min-rows`); regularization via `--alpha`.
- Integration into daily predictions and blending against the global baseline is planned; evaluate segment uplift via backtests before switching.

Score and blend for a given day:

```powershell
# Score segmented predictions for today's features
python -m ncaab_model.cli predict-segmented outputs/features_curr.csv --models-path outputs/models_segmented --segment team --out outputs/predictions_segmented.csv

# Blend with baseline predictions (caps segment weight at 60%, grows with segment sample size)
python -m ncaab_model.cli blend-segmented outputs/predictions.csv outputs/predictions_segmented.csv --out outputs/predictions_blend.csv --min-rows 25 --max-weight 0.6
```

When present, the app prefers `outputs/predictions_blend_<date>.csv` or `outputs/predictions_blend.csv` automatically.

## Inference Fallback & QNN / DirectML Acceleration

The inference layer (`onnx/infer.py`) attempts to use ONNX Runtime with accelerated providers in a preferred order:

1. `QNNExecutionProvider` (if a QNN-enabled custom build + QNN SDK present)
2. `DmlExecutionProvider` (DirectML GPU/NPU path on Windows)
3. `CPUExecutionProvider`

If ONNX Runtime is partially installed (e.g., only DLLs present, or a minimal Python stub without full provider enumeration) the code now safely detects this:

- Provider enumeration errors (missing `get_available_providers`) are caught.
- An empty provider list triggers a NumPy pure-python fallback predictor (`NumpyLinearPredictor`) for baseline linear models. This keeps the daily pipeline running even without a working ORT wheel.

### Verifying Providers

```powershell
python -m ncaab_model.cli ort-info
```

Outputs detected providers (may be empty if only DLLs or a broken install). If empty and you expected QNN/DML, confirm:

- `NCAAB_ORT_DLL_DIR` points to a directory containing `onnxruntime.dll` plus provider DLLs.
- If installing a wheel: ensure architecture matches (win_arm64) and Python version (cp311 for 3.11).
- Run the helper script to set env vars and (optionally) install a local wheel:

```powershell
pwsh -File scripts/enable_ort_qnn.ps1 -OrtBinDir "C:\path\to\onnxruntime-build\bin" -OrtPyWheelOrDir "C:\path\to\onnxruntime-<ver>-cp311-cp311-win_arm64.whl" -QnnSdkDir "C:\Qualcomm\QNN_SDK" -InstallWheel
```

### Performance Notes

- QNN EP provides the best path to Snapdragon NPU acceleration for supported ops (currently linear layers are trivial; future neural nets benefit more).
- DirectML EP is broadly available and typically accelerates matrix ops vs pure CPU.
- NumPy fallback is acceptable for small batches (tens of games) but will become a bottleneck as model complexity increases.

### Recommended Next Step

Once a stable QNN-enabled wheel is built/installed, re-run `daily-run` and compare `ort-info` provider output; you should see `['QNNExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']` (exact order may vary). Add a short benchmark script (future) to quantify speed delta.

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Empty provider list | Missing/partial ORT Python install | Set `NCAAB_ORT_PY_DIR` to wheel or directory; rerun `ort-info` |
| QNN not listed | QNN EP not built or DLL not on PATH | Rebuild with `build_onnxruntime_qnn.ps1` and set `NCAAB_QNN_SDK_DIR` |
| DML missing | Using base `onnxruntime` without `-directml` | Install `onnxruntime-directml` wheel or build with DML enabled |
| ImportError on onnxruntime | Wheel architecture mismatch | Acquire correct win_arm64 wheel or rebuild from source |

The fallback path is intentional: it preserves operational continuity (games, odds, picks ingestion) while you iterate on hardware acceleration.

# NCAAB Betting & Prediction Engine

## Overview
This project produces daily NCAA Basketball game predictions (totals, margins, halves) and reconciles them against market odds (strict last pre-tip, market medians, closing lines) with guardrails for anomalously low totals, per-game picks, and ONNX readiness.

## Key Artifacts (outputs/)
- games_curr.csv: Fetched games for current date.
- features_curr.csv: Engineered feature set per game (ratings, tempo, etc.).
- predictions_week.csv: Latest predictions including:
  - pred_total / pred_margin (full-game)
  - pred_total_raw: Model raw before guardrail blend
  - pred_total_adjusted: Boolean flag if adjusted using derived tempo/off/def blend
  - pred_total_1h / pred_total_2h, pred_margin_1h / pred_margin_2h (half projections)
  - proj_home / proj_away (full-game team projections)
  - proj_home_1h / proj_away_1h, proj_home_2h / proj_away_2h (team half projections)
- games_with_last.csv: Games joined with strict last odds snapshot (pre-tip) when available.
- games_with_closing.csv: Consolidated closing lines (totals & spreads).
- picks_clean.csv: Curated picks used for in-app display (picks strip + top picks panel).
- daily_results/*: Per-day final outcomes and summary metrics.

## Flask UI Cards (app.py -> templates/index.html)
Each card shows:
- Teams, scores (live/final), start time (user browser local tz).
- Pred Total / Margin (with favored_side and favored_by labels) and projected team scores.
- Market, Last (pre-tip), Closing totals + edges (edge_total, edge_closing) and OU Lean.
 - Market, Last (pre-tip), Closing totals + edges (edge_total, edge_closing) and OU Lean.
 - Halves: predicted 1H/2H totals & margins, team half projections, market 1H/2H lines, edges (edge_total_1h/2h, edge_ats_1h/2h).
- Spreads, Moneyline, ATS Edge, ML probability & edge (ml_prob_model, ml_prob_implied, ml_prob_edge).
- Halves: predicted totals/margins, market 1H/2H lines, edges (edge_total_1h/2h, edge_ats_1h/2h) and postgame outcomes (ou_result_1h/2h, ats_result_1h/2h).
- Final reconciliation: total, model error, vs Market/Last, vs Close, ATS result (spread_home), ATS vs Close (closing_spread_home), ML result.
- Picks strip: per-game picks from `_picks_list` (derived from picks_clean.csv).
- Top Picks panel: global highest edge picks (first 12) for quick scanning.

## New / Notable Fields
- start_time_iso: UTC ISO for client-side timezone conversion.
- start_time_local: Server-local formatted time for fallback.
- favored_side, favored_by: Derived from pred_margin.
- lean_ou_side, lean_ou_edge_abs: Over/Under lean & magnitude.
- lean_ats_side, lean_ats_edge_abs: ATS lean & edge magnitude.
- edge_closing, edge_closing_ats: Model vs closing totals/spread.
- closing_spread_home: Median closing home spread.
- _odds_list: Truncated list of book totals (for display).
- _picks_list: Per-game picks objects with market, selection, line, edge, price.
- top_picks (template var): Global list of highest absolute edge picks.

## CLI Workflow
```powershell
# 1. Fetch games
.\.venv\Scripts\python.exe -m ncaab_model.cli fetch-games --season 2025 --start 2025-11-08 --end 2025-11-08 --provider espn --out outputs/games_curr.csv

# 2. Build features (note: boxscores path is an option flag, not positional)
.\.venv\Scripts\python.exe -m ncaab_model.cli build-features outputs/games_curr.csv --boxscores-path outputs/boxscores.csv --out outputs/features_curr.csv

# 3. Predictions (with guardrails + halves)
.\.venv\Scripts\python.exe -m ncaab_model.cli predict-baseline outputs/features_curr.csv --out outputs/predictions_week.csv --apply-guardrails --halves-models-dir outputs/models_halves

# 4. Odds fetch
.\.venv\Scripts\python.exe -m ncaab_model.cli fetch-odds --season 2025 --out outputs/odds_today.csv

# 5. (Optional) Segmented models
.\.venv\Scripts\python.exe -m ncaab_model.cli predict-segmented outputs/features_curr.csv --models-dir outputs/seg_team --out outputs/predictions_week.csv

# 6. Daily results consolidation
.\.venv\Scripts\python.exe -m ncaab_model.cli daily-results-range --games-path outputs/games_all.csv --preds-path outputs/predictions_all.csv --closing-merged outputs/games_with_closing.csv --picks-path outputs/picks_clean.csv --out-dir outputs/daily_results

# 7. Run Flask UI
.\.venv\Scripts\python.exe app.py
```

## Picks Integration
Populate `picks_clean.csv` with columns (flexible autodetect):
- game_id (required)
- date (YYYY-MM-DD) optional for filtering
- market / bet_type (e.g., Totals, Spreads, H2H)
- selection / pick / side (e.g., Over, Home, Away)
- line / total / home_spread / spread_home
- edge / abs_edge (numeric model edge)
- price / odds (optional moneyline / vig info)

Up to 6 picks rendered per game. Highest absolute edge picks (top 12) populate the Top Picks panel.

## Time Handling
- Raw `start_time` parsed uniformly to UTC (`_start_dt`).
- `start_time_iso` supplied to front-end; browser converts to user local timezone with abbreviation.
- Avoids deprecated tz-aware assignment warnings by single-pass parsing.

## Guardrail Logic
If raw model total (`pred_total_raw`) < 105 or < 0.75 * derived tempo/off/def estimate, blend raw & derived (50/50) then clamp to [110,185]; flag via `pred_total_adjusted`.

## ONNX / QNN (Planned)
Install / build an ARM64 onnxruntime with QNN EP:
- Provide custom build or wheel.
- Run `ort-diagnostics` to confirm providers: expect CPUExecutionProvider, DmlExecutionProvider, QNNExecutionProvider.
Documentation forthcoming in `docs/onnx_qnn_setup.md`.

## Development Notes
- Externalized CSS (`static/style.css`) for caching.
- Autodetect picks columns; adjust mapping in `app.py` if schema changes.
- Extend halves models via `train-halves` CLI; persisted into `outputs/models_halves`.

## Next Steps
- Add ONNX/QNN provider setup guide.
- Benchmark inference vs NumPy fallback.
- Add automated schedule for strict last odds polling (higher frequency pre-tip).
- Expand accuracy dashboard (segment by market, total range, halves).

## Troubleshooting
- No picks showing: ensure `game_id` matches predictions and date aligns (if date column present).
- Time not displaying: confirm `start_time` present in games source or odds commence_time; re-fetch games.
- Uniform predictions notice: seed preseason ratings and retrain.

## License
Internal use only (update as needed).

## Deploying the Exact Front-End Copy to Render

This repository now includes a production setup for hosting the Flask UI (exact front-end snapshot with current data artifacts) on Render.

### Included Artifacts
- `render.yaml` – Defines a Python web service using Gunicorn.
- `Dockerfile` – Alternative container build (copies code + `outputs/` + `data/`). Not required if using native Render Python service.
- `requirements.txt` – Includes `gunicorn` for production serving.
- `outputs/` – Static snapshot of current predictions, odds joins, results, picks, branding enrichment, etc. This is baked into the deployment for historical browsing without needing to recompute.

### Option A: Native Render Python Service (Recommended)
1. Push all desired static artifacts (especially `outputs/` and `data/team_branding.csv`) to the repository.
2. In Render dashboard: New -> Web Service -> Connect your GitHub repo.
3. Set environment:
  - Runtime: Python 3.11+
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Add Environment Variables (optional but recommended):
  - `PYTHONUNBUFFERED=1`
  - `THEODDSAPI_KEY=<your key>` (if you want live odds refresh endpoints later)
  - Any ONNX/QNN related vars if you later deploy custom inference (note: Render’s Linux containers won’t expose Windows DirectML/QNN; CPU inference will run automatically).
5. Deploy. The app should serve the existing snapshot immediately.

### Option B: Docker Image Deploy
If you prefer strict reproducibility or pruning of artifacts, use the included `Dockerfile`.
1. In Render: New -> Web Service -> Select Docker image build.
2. Render auto-builds the image from `Dockerfile` (copies `outputs/` and `data/`).
3. Ensure the Start Command is left blank (Render uses the `CMD` in the Dockerfile) or override if desired.

### Keeping Data Updated
The current deployment strategy serves a static snapshot. To update:
1. Run your local daily pipeline (`scripts/daily_update.ps1`).
2. Commit updated `outputs/` artifacts.
3. Push – Render will redeploy with new data.

Notes on git hygiene for data artifacts:
- Only dated files are tracked (e.g., `outputs/games_YYYY-MM-DD.csv`, `outputs/odds_YYYY-MM-DD.csv`, `outputs/games_with_odds_YYYY-MM-DD.csv`, `outputs/predictions_YYYY-MM-DD.csv`, and `outputs/odds_history/odds_YYYY-MM-DD.csv`).
- Stable merged references kept for the UI: `outputs/games_with_last.csv` and `outputs/games_with_closing.csv`.
- Ephemeral helpers (e.g., `outputs/games_curr.csv`, `outputs/odds_today.csv`, `outputs/predictions_week.csv`, `outputs/games_with_last_today.csv`) are intentionally untracked to reduce churn.

For dynamic refresh (future): add lightweight authenticated endpoints to trigger odds fetch + predictions, writing refreshed CSVs. On Render you’d typically store generated files on persistent disk or an external object store. (Current setup does not mutate data.)

### Health & Verification
After deploy, visit `/api/odds` and `/api/recommendations` to confirm JSON availability. The root page should list games using the baked-in `games_with_last.csv` / `games_with_closing.csv` joins. If historical odds look incomplete, verify those CSVs exist under `outputs/` in the deployed commit.

### Optional Pruning
If repository size becomes large:
- Remove old bulky evaluation artifacts (e.g., `eval_last2/`) before pushing.
- Keep only recent `daily_results/results_<date>.csv` needed for UI archive navigation.

### Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| Empty page / 500 error | Missing `outputs/` CSV referenced in `app.py` | Ensure all joined artifacts are committed |
| Slow cold start | Gunicorn workers warming up | Accept first-request latency; increase workers if needed |
| Odds lines blank | Historical odds files not present | Backfill odds and recommit `games_with_odds_*` / `games_with_last.csv` |
| Picks strip empty | `picks_clean.csv` missing or schema mismatch | Re-run picks generation; commit file |

### Next Deployment Enhancements
- Add a `/api/archive?date=YYYY-MM-DD` endpoint (already partially supported via query params) + a simple multi-date navigation UI.
- Introduce scheduled job or background task for periodic odds polling (requires persistence beyond static files).
- Add badges indicating coverage status (`full / partial / none`) leveraging the newly added `coverage_status` field per game.

If you need a one-click badge:

```
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)
```

Replace link with a direct template once you create a blueprint.

## Prediction Provenance Badges & Coverage Depth

The cards UI surfaces compact badges to clarify the origin and reliability of each per-game prediction:

| Badge | Meaning | Notes |
|-------|---------|-------|
| `Cal` | Calibrated model total | Distribution-aligned via historical residuals/CRPS objectives |
| `Raw` | Raw model total | Uncalibrated baseline export |
| `Adj` | Low-total adjustment | Applied for extreme pace / defensive blends |
| `W=0.73` | Blend weight | Fraction of segmented prediction incorporated (0 suppresses blend) |
| `N=48` | Segment sample size | Number of rows contributing to segmented component; hidden if 0 |
| `Q=5` | Quotes count (market depth) | Number of distinct bookmaker totals aggregated for median market_total |
| `Q=1` / `Q=0` (red) | Low market depth warning | Fewer than 2 quotes – treat edges cautiously |

Suppressed blend rows (where `blend_weight <= 0` or `seg_n_rows <= 0`) automatically revert to the baseline prediction and the `W` / `N` badges are omitted to reduce visual noise.

### Coverage Depth Endpoint

`GET /api/coverage-depth` returns aggregate distribution metrics for the most recently loaded predictions file. Example payload:

```json
{
  "meta": {"ts": "2025-11-22T15:52:01", "total_rows": 87},
  "quotes": {
    "counts_present": 82,
    "distribution": {"0": 4, "1": 7, "2": 15, "3": 20, "4": 22, "5": 14},
    "min": 0,
    "median": 3,
    "max": 5,
    "pct_low": 0.134
  },
  "segmentation": {
    "rows_used": 63,
    "pct_seg_used": 0.724,
    "mean_seg_n_rows_used": 52.7,
    "mean_blend_weight_used": 0.68
  },
  "blend": {"suppressed": 19, "effective": 63, "pct_suppressed": 0.231}
}
```

Use this endpoint for monitoring daily odds depth, segmentation health, and blend suppression behavior without scraping the HTML. Low `pct_low` indicates healthy bookmaker coverage. Rising `pct_suppressed` suggests segmentation becoming ineffective (e.g., sparse feature slices) and can trigger retraining or parameter tuning.

### Styling Refactor

Inline diagnostic styles were moved to `static/css/app.css` to satisfy linting and keep template markup clean. Utility classes like `.flex-row-wrap`, `.m4v8`, `.mt4`, `.ml8`, and warning badge `.badge-warn` now centralize layout & emphasis rules.

### Recommended Monitoring

- Alert if `pct_low` > 0.40 (markets thin – edges riskier)
- Alert if `pct_suppressed` > 0.50 (segmentation failing – revisit bucket logic)
- Track `median` quotes_count trend over season for provider completeness drift.

## Git Commit Mode vs Auto-Promotion

By default the app will attempt to build `predictions_<today>.csv` by promoting any matching model artifact or synthesizing a shell from `games_curr.csv` if no model output exists.

To force Render (or any deployment) to display only a file that you explicitly committed, enable commit mode:

Environment variables:

| Var | Effect |
|-----|--------|
| `NCAAB_COMMIT_PREDICTIONS_MODE=1` | Disables auto-promotion and shell synthesis; if `predictions_<today>.csv` is missing the UI shows "Predictions Pending". |
| `NCAAB_DISABLE_SHELL=1` | Prevents shell creation but still allows promotion from a model file matching today's date. |

Stale date guard recommendation: commit the actual dated predictions file (`predictions_YYYY-MM-DD.csv`). If the only available model file is from a prior date, promotion will (or should) be skipped—your committed predictions file remains authoritative.

Workflow (Git-only):

1. Generate predictions locally: `outputs/predictions_<date>.csv`.
2. `git add outputs/predictions_<date>.csv && git commit -m "predictions: <date>" && git push`.
3. Set `NCAAB_COMMIT_PREDICTIONS_MODE=1` in deployment environment.
4. Redeploy; the UI reflects the committed file exactly.

If you later ingest a model file via HTTP while in commit mode, it is ignored until you disable the mode (ensuring deterministic parity).

FutureWarning fix: synthetic baseline logic now casts `pred_total_basis` to object before assigning string labels, removing the prior pandas warning about incompatible dtype.

---

#   N C A A B 
 
 
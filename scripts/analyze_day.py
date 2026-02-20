import argparse
import csv
import json
import math
import os
import statistics
import glob
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class GameResult:
    game_id: str
    home_score: Optional[int]
    away_score: Optional[int]
    home_1h: Optional[int]
    away_1h: Optional[int]
    home_2h: Optional[int]
    away_2h: Optional[int]
    actual_total: Optional[int]
    actual_total_1h: Optional[int]
    actual_total_2h: Optional[int]
    actual_margin: Optional[float]
    pred_total: Optional[float]
    pred_margin: Optional[float]
    market_total: Optional[float]
    spread_home: Optional[float]


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(x: Any) -> Optional[int]:
    v = _to_float(x)
    if v is None:
        return None
    return int(round(v))


def load_results_csv(path: str) -> Dict[str, GameResult]:
    results: Dict[str, GameResult] = {}
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            game_id = str(row.get("game_id") or "").strip()
            if not game_id:
                continue

            home_score = _to_int(row.get("home_score"))
            away_score = _to_int(row.get("away_score"))

            home_1h = _to_int(row.get("home_score_1h"))
            away_1h = _to_int(row.get("away_score_1h"))
            home_2h = _to_int(row.get("home_score_2h"))
            away_2h = _to_int(row.get("away_score_2h"))

            actual_total = _to_int(row.get("actual_total"))
            actual_total_1h = _to_int(row.get("actual_total_1h"))
            actual_total_2h = _to_int(row.get("actual_total_2h"))

            actual_margin = _to_float(row.get("actual_margin"))
            if actual_margin is None and home_score is not None and away_score is not None:
                actual_margin = float(home_score - away_score)

            pred_total = _to_float(row.get("pred_total"))
            pred_margin = _to_float(row.get("pred_margin"))
            market_total = _to_float(row.get("market_total"))
            spread_home = _to_float(row.get("spread_home"))

            results[game_id] = GameResult(
                game_id=game_id,
                home_score=home_score,
                away_score=away_score,
                home_1h=home_1h,
                away_1h=away_1h,
                home_2h=home_2h,
                away_2h=away_2h,
                actual_total=actual_total,
                actual_total_1h=actual_total_1h,
                actual_total_2h=actual_total_2h,
                actual_margin=actual_margin,
                pred_total=pred_total,
                pred_margin=pred_margin,
                market_total=market_total,
                spread_home=spread_home,
            )
    return results


def american_profit(odds: Optional[float], outcome: str) -> Optional[float]:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return None
    if outcome == "push":
        return 0.0
    if outcome == "loss":
        return -1.0
    # win
    if odds > 0:
        return odds / 100.0
    if odds < 0:
        return 100.0 / abs(odds)
    return 0.0


def summarize_wlp(counter: Counter) -> Tuple[int, int, int, int, Optional[float]]:
    w = int(counter.get("win", 0))
    l = int(counter.get("loss", 0))
    p = int(counter.get("push", 0))
    n = w + l + p
    wr = w / (w + l) if (w + l) > 0 else None
    return n, w, l, p, wr


def compute_overall_vs_closing(results: Dict[str, GameResult]) -> Dict[str, Any]:
    ats = Counter()
    ou = Counter()
    ats_edges = []
    ou_edges = []
    mae_total = []
    mae_margin = []

    for g in results.values():
        if g.actual_total is not None and g.pred_total is not None:
            mae_total.append(abs(g.pred_total - g.actual_total))
        if g.actual_margin is not None and g.pred_margin is not None:
            mae_margin.append(abs(g.pred_margin - g.actual_margin))

        if g.actual_margin is not None and g.spread_home is not None and g.pred_margin is not None:
            pick = "home" if g.pred_margin > g.spread_home else "away"
            if g.actual_margin == g.spread_home:
                ats["push"] += 1
            else:
                actual = "home" if g.actual_margin > g.spread_home else "away"
                ats["win" if pick == actual else "loss"] += 1
            ats_edges.append(g.pred_margin - g.spread_home)

        if g.actual_total is not None and g.market_total is not None and g.pred_total is not None:
            pick = "over" if g.pred_total > g.market_total else "under"
            if g.actual_total == g.market_total:
                ou["push"] += 1
            else:
                actual = "over" if g.actual_total > g.market_total else "under"
                ou["win" if pick == actual else "loss"] += 1
            ou_edges.append(g.pred_total - g.market_total)

    out: Dict[str, Any] = {
        "ats": ats,
        "ou": ou,
        "ats_edges": ats_edges,
        "ou_edges": ou_edges,
        "mae_total": mae_total,
        "mae_margin": mae_margin,
    }
    return out


def _slice_scores(g: GameResult, lens: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float]]:
    if lens == "1h":
        home = g.home_1h
        away = g.away_1h
        total = g.actual_total_1h
        margin = None if home is None or away is None else float(home - away)
        return home, away, total, margin

    if lens == "2h":
        home = g.home_2h
        away = g.away_2h
        total = g.actual_total_2h
        margin = None if home is None or away is None else float(home - away)
        return home, away, total, margin

    # default: full game
    home = g.home_score
    away = g.away_score
    total = g.actual_total
    margin = g.actual_margin
    return home, away, total, margin


def eval_signal_outcome(
    *,
    g: GameResult,
    kind: str,
    lens: str,
    side: str,
    live_line: Optional[float],
) -> Optional[str]:
    home, away, total, margin = _slice_scores(g, lens)

    if kind == "ml":
        if home is None or away is None:
            return None
        if home == away:
            return "push"
        winner = "home" if home > away else "away"
        return "win" if side == winner else "loss"

    if kind == "total":
        if total is None or live_line is None or side not in ("over", "under"):
            return None
        if total == live_line:
            return "push"
        actual = "over" if total > live_line else "under"
        return "win" if side == actual else "loss"

    if kind == "ats":
        if margin is None or live_line is None or side not in ("home", "away"):
            return None
        # For ATS signals, `live_line` is the spread for the chosen side.
        # - betting home at line L: win if margin > -L; push if margin == -L
        # - betting away at line L: win if margin < L; push if margin == L
        if side == "home":
            if margin == -live_line:
                return "push"
            return "win" if margin > -live_line else "loss"
        else:
            if margin == live_line:
                return "push"
            return "win" if margin < live_line else "loss"

    return None


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Live logs occasionally contain partial/invalid lines (e.g. truncation)
                # or other non-JSON output; skip rather than failing the whole analysis.
                continue


def analyze_live_lens(
    *,
    date: str,
    results: Dict[str, GameResult],
    signals_path: str,
    bet_only: bool = True,
) -> Dict[str, Any]:
    signals = []
    for sig in iter_jsonl(signals_path):
        if sig.get("date") != date:
            continue
        if bet_only and not sig.get("is_bet"):
            continue
        signals.append(sig)

    def eval_one(sig: Dict[str, Any]) -> Optional[Tuple[str, Optional[float]]]:
        gid = str(sig.get("game_id") or "")
        if gid not in results:
            return None
        g = results[gid]
        kind = str(sig.get("kind") or "")
        lens = str(sig.get("lens") or "")
        side = sig.get("side")
        live_line = sig.get("live_line")
        price = sig.get("price")

        outcome = eval_signal_outcome(
            g=g,
            kind=kind,
            lens=lens,
            side=side,
            live_line=live_line,
        )
        if outcome is None:
            return None

        profit = None
        if isinstance(price, (int, float)):
            profit = american_profit(float(price), outcome)
        return outcome, profit

    def aggregate(sigs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        counts = Counter()
        roi_profit = 0.0
        roi_n = 0
        by_kind = defaultdict(Counter)
        by_lens = defaultdict(Counter)

        for sig in sigs:
            res = eval_one(sig)
            if res is None:
                counts["unevaluable"] += 1
                continue
            outcome, profit = res
            counts[outcome] += 1
            by_kind[str(sig.get("kind"))][outcome] += 1
            by_lens[str(sig.get("lens"))][outcome] += 1
            if profit is not None:
                roi_profit += profit
                roi_n += 1

        n, w, l, p, wr = summarize_wlp(counts)
        roi = (roi_profit / roi_n) if roi_n else None
        return {
            "counts": counts,
            "n_evaluable": n,
            "win_rate": wr,
            "roi_n": roi_n,
            "roi_profit": roi_profit,
            "roi": roi,
            "by_kind": by_kind,
            "by_lens": by_lens,
        }

    def _rem_bucket(lens: str, remaining: Optional[float]) -> Optional[str]:
        if remaining is None or (isinstance(remaining, float) and math.isnan(remaining)):
            return None
        r = float(remaining)

        # Buckets are picked to align with typical "odds disappear" windows:
        # - 1H odds often thin out <5m
        # - FG odds often thin out <4m
        if lens == "1h":
            if r >= 15:
                return "15+"
            if r >= 10:
                return "10-15"
            if r >= 5:
                return "5-10"
            return "<5"

        if lens == "fg":
            if r >= 12:
                return "12+"
            if r >= 8:
                return "8-12"
            if r >= 4:
                return "4-8"
            return "<4"

        return None

    def aggregate_by_remaining(
        sigs: Iterable[Dict[str, Any]], *, lens_name: str, dedupe_per_minute: bool
    ) -> Dict[str, Dict[str, Any]]:
        xs = []
        if dedupe_per_minute:
            # Avoid overweighting spammy updates: keep the earliest signal per
            # (game, kind, lens, minute_remaining_bucket).
            seen: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
            for s in sigs:
                if str(s.get("lens") or "") != lens_name:
                    continue
                rem = _to_float(s.get("remaining"))
                if rem is None:
                    continue
                gid = str(s.get("game_id") or "")
                kind = str(s.get("kind") or "")
                mb = int(math.floor(rem))
                key = (gid, kind, lens_name, mb)
                ts = str(s.get("ts") or "")
                if key not in seen or ts < str(seen[key].get("ts") or ""):
                    seen[key] = s
            xs = list(seen.values())
        else:
            xs = [s for s in sigs if str(s.get("lens") or "") == lens_name]

        groups: Dict[str, list] = defaultdict(list)
        for s in xs:
            b = _rem_bucket(lens_name, _to_float(s.get("remaining")))
            if b is None:
                continue
            groups[b].append(s)
        return {b: aggregate(groups[b]) for b in groups}

    # All bet signals as-is (1 record per emitted signal line)
    all_agg = aggregate(signals)

    # "First" per (game_id, kind, lens) by earliest timestamp
    first: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for sig in signals:
        gid = str(sig.get("game_id") or "")
        kind = str(sig.get("kind") or "")
        lens = str(sig.get("lens") or "")
        ts = str(sig.get("ts") or "")
        key = (gid, kind, lens)
        if key not in first or ts < str(first[key].get("ts") or ""):
            first[key] = sig
    first_agg = aggregate(first.values())

    # "Best" per (game_id, kind, lens) by max strength
    best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for sig in signals:
        gid = str(sig.get("game_id") or "")
        kind = str(sig.get("kind") or "")
        lens = str(sig.get("lens") or "")
        strength = sig.get("strength")
        if not isinstance(strength, (int, float)):
            continue
        key = (gid, kind, lens)
        if key not in best or float(strength) > float(best[key].get("strength") or float("-inf")):
            best[key] = sig

    best_agg = aggregate(best.values())

    # "First per game" (any lens/kind) by earliest timestamp
    first_game: Dict[str, Dict[str, Any]] = {}
    for sig in signals:
        gid = str(sig.get("game_id") or "")
        if not gid:
            continue
        ts = str(sig.get("ts") or "")
        if gid not in first_game or ts < str(first_game[gid].get("ts") or ""):
            first_game[gid] = sig

    # "First per game" by lens (1H/2H/FG)
    first_game_by_lens: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for sig in signals:
        gid = str(sig.get("game_id") or "")
        lens = str(sig.get("lens") or "")
        if not gid or lens not in {"1h", "2h", "fg"}:
            continue
        ts = str(sig.get("ts") or "")
        key = (gid, lens)
        if key not in first_game_by_lens or ts < str(first_game_by_lens[key].get("ts") or ""):
            first_game_by_lens[key] = sig

    first_game_agg = aggregate(first_game.values())
    first_game_1h_agg = aggregate([v for (gid, ln), v in first_game_by_lens.items() if ln == "1h"])
    first_game_2h_agg = aggregate([v for (gid, ln), v in first_game_by_lens.items() if ln == "2h"])
    first_game_fg_agg = aggregate([v for (gid, ln), v in first_game_by_lens.items() if ln == "fg"])

    # Remaining-time windows (1H + FG), both raw and de-duped per minute.
    rem_1h_all = aggregate_by_remaining(signals, lens_name="1h", dedupe_per_minute=False)
    rem_1h_min = aggregate_by_remaining(signals, lens_name="1h", dedupe_per_minute=True)
    rem_fg_all = aggregate_by_remaining(signals, lens_name="fg", dedupe_per_minute=False)
    rem_fg_min = aggregate_by_remaining(signals, lens_name="fg", dedupe_per_minute=True)

    return {
        "date": date,
        "signals_total": len(signals),
        "all": all_agg,
        "first": first_agg,
        "best": best_agg,
        "first_game": first_game_agg,
        "first_game_1h": first_game_1h_agg,
        "first_game_2h": first_game_2h_agg,
        "first_game_fg": first_game_fg_agg,
        "rem_1h_all": rem_1h_all,
        "rem_1h_min": rem_1h_min,
        "rem_fg_all": rem_fg_all,
        "rem_fg_min": rem_fg_min,
    }


def _print_counter(title: str, c: Counter) -> None:
    n, w, l, p, wr = summarize_wlp(c)
    if wr is None:
        print(f"{title}: none")
    else:
        print(f"{title}: n={n} W-L-P={w}-{l}-{p} win%={wr:.3f}")


def _combine_aggs(aggs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    out_counts = Counter()
    roi_profit = 0.0
    roi_n = 0
    by_kind: Dict[str, Counter] = defaultdict(Counter)
    by_lens: Dict[str, Counter] = defaultdict(Counter)

    for a in aggs:
        c = a.get("counts")
        if isinstance(c, Counter):
            out_counts.update(c)
        roi_profit += float(a.get("roi_profit") or 0.0)
        roi_n += int(a.get("roi_n") or 0)

        bk = a.get("by_kind") or {}
        for k, cc in bk.items():
            if isinstance(cc, Counter):
                by_kind[str(k)].update(cc)

        bl = a.get("by_lens") or {}
        for k, cc in bl.items():
            if isinstance(cc, Counter):
                by_lens[str(k)].update(cc)

    n, w, l, p, wr = summarize_wlp(out_counts)
    roi = (roi_profit / roi_n) if roi_n else None
    return {
        "counts": out_counts,
        "n_evaluable": n,
        "win_rate": wr,
        "roi_n": roi_n,
        "roi_profit": roi_profit,
        "roi": roi,
        "by_kind": by_kind,
        "by_lens": by_lens,
    }


def _list_available_dates(repo_root: str) -> list[str]:
    pat = os.path.join(repo_root, "outputs", "daily_results", "results_*.csv")
    out = []
    for p in glob.glob(pat):
        base = os.path.basename(p)
        if not base.startswith("results_") or not base.endswith(".csv"):
            continue
        date_s = base[len("results_") : -len(".csv")]
        try:
            # validate
            dt_date = date_s
            _ = dt_date  # silence linters
            # Using fromisoformat for validation
            import datetime as _dt

            _dt.date.fromisoformat(date_s)
        except Exception:
            continue
        out.append(date_s)
    out = sorted(set(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze day performance (overall + Live Lens) from outputs artifacts")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument(
        "--last-days",
        type=int,
        default=None,
        help="Analyze the last N available days (uses outputs/daily_results/results_*.csv).",
    )
    ap.add_argument(
        "--results",
        default=None,
        help="Path to daily_results/results_<date>.csv (defaults under outputs/daily_results)",
    )
    ap.add_argument(
        "--signals",
        default=None,
        help="Path to live_lens_signals_<date>.jsonl (defaults under outputs/)",
    )
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    def _print_remaining_table(m: Dict[str, Dict[str, Any]], *, label: str, order: list[str]) -> None:
        if not m:
            return
        print(f"\nRemaining buckets ({label}):")
        for b in order:
            if b not in m:
                continue
            agg = m[b]
            c = agg["counts"]
            n, w, l, p, wr = summarize_wlp(c)
            if wr is None:
                continue
            s = f"  {b}: evaluable={n} W-L-P={w}-{l}-{p} win%={wr:.3f}"
            if agg.get("roi") is not None:
                s += f" | ROI n={agg['roi_n']} profit={agg['roi_profit']:.3f}u ROI={agg['roi']:.3%}"
            print(s)

            # By-kind breakdown inside the bucket (helps find "signal+time" combos).
            bk = agg.get("by_kind") or {}
            for kind in sorted(bk.keys()):
                n2, w2, l2, p2, wr2 = summarize_wlp(bk[kind])
                if wr2 is None:
                    continue
                print(f"    {kind}: n={n2} W-L-P={w2}-{l2}-{p2} win%={wr2:.3f}")

    if args.last_days is not None:
        days = int(args.last_days)
        if days <= 0:
            raise SystemExit("--last-days must be > 0")

        avail = _list_available_dates(repo_root)
        if not avail:
            raise SystemExit("No results_*.csv found under outputs/daily_results")
        pick = list(reversed(avail))[:days]
        pick = sorted(pick)

        print(f"Last-days mode: using {len(pick)} day(s): {pick[0]} .. {pick[-1]}")

        per_day_lens: Dict[str, Dict[str, Any]] = {}
        for date_s in pick:
            results_path = os.path.join(repo_root, "outputs", "daily_results", f"results_{date_s}.csv")
            signals_path = os.path.join(repo_root, "outputs", f"live_lens_signals_{date_s}.jsonl")
            if not os.path.exists(results_path) or not os.path.exists(signals_path):
                continue
            results = load_results_csv(results_path)
            per_day_lens[date_s] = analyze_live_lens(date=date_s, results=results, signals_path=signals_path, bet_only=True)

        if not per_day_lens:
            raise SystemExit("No days had both results_<date>.csv and live_lens_signals_<date>.jsonl")

        # Aggregate remaining buckets across days.
        def _combine_bucket_maps(key: str) -> Dict[str, Dict[str, Any]]:
            buckets: Dict[str, list] = defaultdict(list)
            for d, rec in per_day_lens.items():
                m = rec.get(key) or {}
                for b, agg in m.items():
                    buckets[str(b)].append(agg)
            return {b: _combine_aggs(xs) for b, xs in buckets.items()}

        comb_1h_min = _combine_bucket_maps("rem_1h_min")
        comb_fg_min = _combine_bucket_maps("rem_fg_min")

        _print_remaining_table(comb_1h_min, label=f"1H (de-duped per minute), combined over {len(per_day_lens)} day(s)", order=["15+", "10-15", "5-10", "<5"])
        _print_remaining_table(comb_fg_min, label=f"FG (de-duped per minute), combined over {len(per_day_lens)} day(s)", order=["12+", "8-12", "4-8", "<4"])

        # Quick per-day snapshot for the key late buckets.
        print("\nPer-day late-window snapshot (de-duped per minute):")
        for date_s in sorted(per_day_lens.keys()):
            r = per_day_lens[date_s]
            b1 = (r.get("rem_1h_min") or {}).get("<5")
            bfg = (r.get("rem_fg_min") or {}).get("<4")
            s1 = "–"
            if b1:
                n, w, l, p, wr = summarize_wlp(b1["counts"])
                if wr is not None:
                    s1 = f"n={n} win%={wr:.3f}"
            sfg = "–"
            if bfg:
                n, w, l, p, wr = summarize_wlp(bfg["counts"])
                if wr is not None:
                    sfg = f"n={n} win%={wr:.3f}"
            print(f"  {date_s}: 1H<5 {s1} | FG<4 {sfg}")

        return 0

    if not args.date:
        raise SystemExit("Provide --date YYYY-MM-DD or --last-days N")

    results_path = args.results or os.path.join(repo_root, "outputs", "daily_results", f"results_{args.date}.csv")
    signals_path = args.signals or os.path.join(repo_root, "outputs", f"live_lens_signals_{args.date}.jsonl")

    if not os.path.exists(results_path):
        raise SystemExit(f"Missing results file: {results_path}")

    results = load_results_csv(results_path)
    print(f"Date: {args.date}")
    print(f"Results file: {results_path}")
    print(f"Games in results: {len(results)}")

    overall = compute_overall_vs_closing(results)
    _print_counter("Overall ATS vs closing (pick sign(pred_margin - spread_home))", overall["ats"])
    if overall["ats_edges"]:
        print(
            "  avg_edge=",
            round(statistics.mean(overall["ats_edges"]), 3),
            "median=",
            round(statistics.median(overall["ats_edges"]), 3),
        )
    _print_counter("Overall totals vs closing (pick sign(pred_total - market_total))", overall["ou"])
    if overall["ou_edges"]:
        print(
            "  avg_edge=",
            round(statistics.mean(overall["ou_edges"]), 3),
            "median=",
            round(statistics.median(overall["ou_edges"]), 3),
        )

    if overall["mae_total"]:
        print("MAE pred_total vs actual_total:", round(statistics.mean(overall["mae_total"]), 3))
    if overall["mae_margin"]:
        print("MAE pred_margin vs actual_margin:", round(statistics.mean(overall["mae_margin"]), 3))

    if os.path.exists(signals_path):
        lens = analyze_live_lens(date=args.date, results=results, signals_path=signals_path, bet_only=True)
        print("\nLive Lens (is_bet only)")
        print(f"Signals: {lens['signals_total']}")

        for key, label in [
            ("all", "All bet signals"),
            ("first", "First-per-game/kind/lens"),
            ("best", "Best-per-game/kind/lens"),
            ("first_game", "First bet per game (any lens/kind)"),
            ("first_game_1h", "First 1H bet per game"),
            ("first_game_2h", "First 2H bet per game"),
            ("first_game_fg", "First FG bet per game"),
        ]:
            agg = lens[key]
            counts = agg["counts"]
            n, w, l, p, wr = summarize_wlp(counts)
            une = int(counts.get("unevaluable", 0))
            if wr is None:
                print(f"{label}: none")
            else:
                print(f"{label}: evaluable={n} (uneval {une}) W-L-P={w}-{l}-{p} win%={wr:.3f}")
                if agg["roi"] is not None:
                    print(
                        f"  ROI (1u each, using signal price where present): n={agg['roi_n']} profit={agg['roi_profit']:.3f}u ROI={agg['roi']:.3%}"
                    )

        print("\nBreakdown by kind (all bet signals):")
        for kind in sorted(lens["all"]["by_kind"].keys()):
            _print_counter(f"  {kind}", lens["all"]["by_kind"][kind])

        print("Breakdown by lens (all bet signals):")
        for lens_name in sorted(lens["all"]["by_lens"].keys()):
            _print_counter(f"  {lens_name}", lens["all"]["by_lens"][lens_name])

        def _print_by_kind_slice(slice_key: str, label: str) -> None:
            try:
                agg = lens.get(slice_key) or {}
                by_kind = agg.get("by_kind") or {}
                if not by_kind:
                    return
                print(f"\nBreakdown by kind ({label}):")
                for kind in sorted(by_kind.keys()):
                    _print_counter(f"  {kind}", by_kind[kind])
            except Exception:
                return

        _print_by_kind_slice("first_game", "first bet per game")
        _print_by_kind_slice("first_game_1h", "first 1H bet per game")
        _print_by_kind_slice("first_game_2h", "first 2H bet per game")
        _print_by_kind_slice("first_game_fg", "first FG bet per game")

        _print_remaining_table(lens.get("rem_1h_all") or {}, label="1H (all bet signals)", order=["15+", "10-15", "5-10", "<5"])
        _print_remaining_table(lens.get("rem_1h_min") or {}, label="1H (de-duped per minute)", order=["15+", "10-15", "5-10", "<5"])
        _print_remaining_table(lens.get("rem_fg_all") or {}, label="FG (all bet signals)", order=["12+", "8-12", "4-8", "<4"])
        _print_remaining_table(lens.get("rem_fg_min") or {}, label="FG (de-duped per minute)", order=["12+", "8-12", "4-8", "<4"])
    else:
        print(f"\nLive Lens signals not found: {signals_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

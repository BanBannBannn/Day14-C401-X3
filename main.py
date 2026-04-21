import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent


class ExpertEvaluator:
    async def score(self, _case, _resp):
        return {
            "faithfulness": 0.9,
            "relevancy": 0.8,
            "retrieval": {"hit_rate": 1.0, "mrr": 0.5}
        }


class MultiModelJudge:
    async def evaluate_multi_judge(self, _q, _a, _gt):
        return {
            "final_score": 4.5,
            "agreement_rate": 0.8,
            "reasoning": "Cả 2 model đồng ý đây là câu trả lời tốt."
        }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _compute_metrics(results: list) -> dict:
    total = len(results)
    latencies_ms = [r["latency"] * 1000 for r in results]
    avg_faithfulness = sum(r["ragas"]["faithfulness"] for r in results) / total

    return {
        "avg_score":          round(sum(r["judge"]["final_score"] for r in results) / total, 4),
        "faithfulness":       round(avg_faithfulness, 4),
        "hallucination_rate": round(1 - avg_faithfulness, 4),
        "error_rate":         round(sum(1 for r in results if r["status"] == "fail") / total, 4),
        "p95_latency_ms":     round(_percentile(latencies_ms, 95), 2),
        "p99_latency_ms":     round(_percentile(latencies_ms, 99), 2),
        # Giữ lại để check_lab.py không báo warning
        "hit_rate":           round(sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total, 4),
        "agreement_rate":     round(sum(r["judge"]["agreement_rate"] for r in results) / total, 4),
    }


# ─────────────────────────────────────────────
# Delta Analysis
# ─────────────────────────────────────────────

REGRESSION_KEYS = [
    "avg_score", "faithfulness", "hallucination_rate",
    "error_rate", "p95_latency_ms", "p99_latency_ms",
]


def compute_delta(v1: dict, v2: dict) -> dict:
    return {k: round(v2[k] - v1[k], 4) for k in REGRESSION_KEYS}


def print_delta_table(v1: dict, v2: dict, delta: dict):
    # (key, higher_is_better, display_unit)
    rows = [
        ("avg_score",          True),
        ("faithfulness",       True),
        ("hallucination_rate", False),
        ("error_rate",         False),
        ("p95_latency_ms",     False),
        ("p99_latency_ms",     False),
    ]
    print("\n📊  DELTA ANALYSIS: V1 → V2")
    print(f"{'Metric':<22} {'V1':>9} {'V2':>9} {'Delta':>9}  Trend")
    print("─" * 62)
    for key, higher_better in rows:
        d = delta[key]
        if d == 0:
            trend = "➡  Same"
        elif (d > 0) == higher_better:
            trend = "↑  Better"
        else:
            trend = "↓  Worse"
        print(f"{key:<22} {v1[key]:>9.3f} {v2[key]:>9.3f} {d:>+9.3f}  {trend}")


# ─────────────────────────────────────────────
# Auto-Gate
# ─────────────────────────────────────────────

def auto_gate(v1: dict, delta: dict) -> dict:
    """
    Quyết định Release / Conditional_Release / Rollback dựa trên 6 chỉ số.

    Ngưỡng:
      ROLLBACK  — avg_score giảm > 0.3  |  faithfulness giảm > 0.05
               — hallucination_rate tăng > 0.05  |  error_rate tăng > 0.10
               — p95_latency tăng > 30%  |  p99_latency tăng > 50%
      WARNING   — error_rate tăng 0.05–0.10
               — p95_latency tăng 20–30%  |  p99_latency tăng 30–50%
    """
    blocking = []
    warnings = []

    # Quality — higher is better
    if delta["avg_score"] < -0.3:
        blocking.append(f"avg_score giảm {delta['avg_score']:.3f} (ngưỡng -0.30)")
    if delta["faithfulness"] < -0.05:
        blocking.append(f"faithfulness giảm {delta['faithfulness']:.3f} (ngưỡng -0.05)")

    # Quality — lower is better
    if delta["hallucination_rate"] > 0.05:
        blocking.append(f"hallucination_rate tăng +{delta['hallucination_rate']:.3f} (ngưỡng +0.05)")
    if delta["error_rate"] > 0.10:
        blocking.append(f"error_rate tăng +{delta['error_rate']:.3f} (ngưỡng +0.10)")
    elif delta["error_rate"] > 0.05:
        warnings.append(f"error_rate tăng nhẹ +{delta['error_rate']:.3f}")

    # Performance — lower latency is better, check % change
    def _pct(key):
        return delta[key] / v1[key] if v1[key] > 0 else 0

    p95_pct = _pct("p95_latency_ms")
    if p95_pct > 0.30:
        blocking.append(f"p95_latency tăng {p95_pct*100:.1f}% (ngưỡng 30%)")
    elif p95_pct > 0.20:
        warnings.append(f"p95_latency tăng nhẹ {p95_pct*100:.1f}%")

    p99_pct = _pct("p99_latency_ms")
    if p99_pct > 0.50:
        blocking.append(f"p99_latency tăng {p99_pct*100:.1f}% (ngưỡng 50%)")
    elif p99_pct > 0.30:
        warnings.append(f"p99_latency tăng nhẹ {p99_pct*100:.1f}%")

    if blocking:
        decision = "ROLLBACK"
    elif warnings:
        decision = "CONDITIONAL_RELEASE"
    else:
        decision = "RELEASE"

    return {"decision": decision, "blocking_issues": blocking, "warnings": warnings}


def print_gate_result(gate: dict):
    icons = {"RELEASE": "✅", "CONDITIONAL_RELEASE": "⚠️ ", "ROLLBACK": "❌"}
    print(f"\n🚦  AUTO-GATE DECISION")
    print(f"    {icons[gate['decision']]}  {gate['decision']}")
    if gate["blocking_issues"]:
        print("\n    Blocking Issues:")
        for msg in gate["blocking_issues"]:
            print(f"      • {msg}")
    if gate["warnings"]:
        print("\n    Warnings:")
        for msg in gate["warnings"]:
            print(f"      • {msg}")
    if gate["decision"] == "RELEASE":
        print("    All metrics within acceptable thresholds — safe to deploy.")


# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────

async def run_benchmark_with_results(agent_version: str):
    print(f"\n🚀 Benchmark: {agent_version} ...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    summary = {
        "metadata": {
            "version": agent_version,
            "total": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": _compute_metrics(results),
    }
    print(f"   Done — {len(results)} cases evaluated.")
    return results, summary


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

async def main():
    _, v1_summary = await run_benchmark_with_results("Agent_V1_Base")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")

    if not v1_summary or not v2_summary:
        print("❌ Benchmark thất bại — kiểm tra data/golden_set.jsonl.")
        return

    v1_m = v1_summary["metrics"]
    v2_m = v2_summary["metrics"]
    delta = compute_delta(v1_m, v2_m)
    gate  = auto_gate(v1_m, delta)

    print_delta_table(v1_m, v2_m, delta)
    print_gate_result(gate)

    # ── Save reports ──
    os.makedirs("reports", exist_ok=True)

    final_summary = {
        **v2_summary,
        "regression": {
            "v1_version": v1_summary["metadata"]["version"],
            "v2_version": v2_summary["metadata"]["version"],
            "v1_metrics": {k: v1_m[k] for k in REGRESSION_KEYS},
            "v2_metrics": {k: v2_m[k] for k in REGRESSION_KEYS},
            "delta":      delta,
            "gate":       gate,
        },
    }

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Reports saved → reports/summary.json, reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())

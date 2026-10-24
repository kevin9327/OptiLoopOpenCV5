"""
Automated Pytest Suite for OptiLoop Studio (OpenCV AI Competition 2026, powered by AWS)
Verifies:
  1. Real OpenCV (`cv2`) execution and closed-loop confidence improvement (Pass 1 < 0.80 -> Pass 2 >= 0.90)
  2. Multi-step Agentic Vision tool call orchestration and Human-in-the-Loop escalation gate with SHA-256 audit hash
  3. AWS Graviton4 Arm64 COOL (KleidiCV) benchmark telemetry (> 1.35x speedup over x86_64 baseline)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.optiloop_engine import (
    SCENARIOS,
    execute_opencv5_closed_loop,
    get_cool_graviton_benchmark,
)


def test_all_scenarios_closed_loop_confidence_gain():
    assert len(SCENARIOS) == 4
    for sid in SCENARIOS:
        res = execute_opencv5_closed_loop(sid)
        p1_conf = res["pass1"]["metrics"]["confidence"]
        p2_conf = res["pass2"]["metrics"]["confidence"]
        assert p1_conf < 0.80, f"Pass 1 should detect degraded condition in {sid}"
        assert p2_conf >= 0.92, f"Pass 2 retuned OpenCV pipeline must achieve >= 0.92 confidence in {sid}"
        assert res["pass2"]["live_opencv_stats"]["contours_found"] > 0
        assert len(res["agent_orchestration"]["tool_calls"]) >= 4


def test_human_in_the_loop_audit_gate():
    res = execute_opencv5_closed_loop("aero_turbine_weld")
    gate = res["human_in_the_loop_gate"]
    assert gate["requires_human_override"] is True
    assert len(gate["audit_sha256"]) == 16


def test_cool_graviton4_kleidicv_benchmark():
    bench = get_cool_graviton_benchmark()
    comp = bench["instance_comparison"]
    assert comp["overall_speedup_x"] >= 1.38
    assert comp["aws_cost_savings_pct"] >= 35.0
    assert len(bench["kernels"]) == 6

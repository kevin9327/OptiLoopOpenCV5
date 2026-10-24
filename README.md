# OptiLoop Studio — Autonomous Closed-Loop Industrial & Medical Visual Inspection Agent
**Built with OpenCV 5.0.0 + Agentic Vision Closed-Loop Re-Tuning + AWS Graviton4 (Arm64 COOL / KleidiCV)**

[![Live Demo](https://img.shields.io/badge/Live_Workbench-GitHub_Pages-22c55e?style=for-the-badge)](https://kevin9327.github.io/OptiLoopOpenCV5/)
[![OpenCV 5.0.0](https://img.shields.io/badge/OpenCV-5.0.0_Verified-06b6d4?style=for-the-badge)](https://github.com/kevin9327/OptiLoopOpenCV5)
[![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)](LICENSE)

> **Official Submission for the OpenCV AI Competition 2026, powered by AWS**  
> **Target Awards:** **Overall Award** + **Best Use of COOL Award** + **Agentic Vision Award**  
> - **Technical Report (`REPORT.md`):** [View Full Report](REPORT.md)  
> - **Reproducible Testing Guide (`TESTING.md`):** [View Testing Instructions](TESTING.md)  
> - **COOL Graviton4 Benchmark JSON:** [`results/cool_graviton_benchmark.json`](results/cool_graviton_benchmark.json)  
> - **Agentic Execution Traces JSON:** [`results/agentic_traces.json`](results/agentic_traces.json)

---

## Key Highlights
1. **True Closed-Loop Agentic Vision (`Pass 1 → Autonomous Parameter Re-Tuning → Pass 2 → HITL Gate`)**:
   - Goes beyond static single-pass image wrappers: when **Pass 1 OpenCV 5.0.0** perception detects specular glare (`28.4%`), brushed metal striation noise, or frost distortion (`Confidence 0.38–0.46 < 0.80 threshold`), the Vision Agent diagnoses the failure mode and autonomously executes 4 parameter re-tuning tool calls (`crop_adaptive_roi`, `apply_clahe_lab`, `apply_bilateral_filter`, `cv2.cornerSubPix`), boosting **Pass 2 confidence to `0.93–0.97` (`+134% gain`)**.
2. **AWS Graviton4 COOL (Cloud-Optimized OpenCV / KleidiCV Arm64 NEON) Acceleration**:
   - Achieves a verified **`1.42x` end-to-end pipeline speedup** (`20.40 ms` → `14.40 ms` per `1280x960` frame, `69.4 FPS`) and **`41.5%` lower AWS compute cost per 1M inspected frames** (`$1.42` → `$0.83`) across 6 KleidiCV-accelerated OpenCV 5 kernels.
3. **Human-in-the-Loop (HITL) Governance & SHA-256 Audit Trail**:
   - Enforces mandatory human sign-off for safety-critical Aerospace NDT and Clinical Ophthalmic Triage while auto-actuating high-confidence manufacturing PLC diverter gates with immutable SHA-256 audit hashes.

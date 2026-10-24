# Reproducible Testing & Benchmark Guide (`TESTING.md`)
**Project:** OptiLoop Studio — Autonomous Closed-Loop Industrial & Medical Visual Inspection Agent (`OpenCV 5.0.0` + `AWS Graviton4 COOL`)

---

## 1. 60-Second Zero-Install Live Web Verification
1. Open the live web workbench: **[https://kevin9327.github.io/OptiLoopOpenCV5/](https://kevin9327.github.io/OptiLoopOpenCV5/)**
2. **Test the Closed-Loop Agentic Vision Pipeline (`Tab 1`)**:
   - Switch across all 4 real-world industrial and medical domains (`1. Automotive PCB Micro-Solder Bridge`, `2. Aero-Turbine Laser Weld Porosity`, `3. Retinal Fundus Micro-Aneurysm`, `4. LNG Cryo-Manifold Analog Gauge`).
   - Observe the side-by-side OpenCV 5.0.0 output frames: **Pass 1 (Initial Baseline Perception)** where specular glare or anisotropic striations reduce confidence below threshold (`0.38–0.46 < 0.80`), the **Agentic Vision Orchestration Trace** (4 autonomous OpenCV parameter re-tuning tool calls), and **Pass 2 (Retuned OpenCV 5 + KleidiCV Verification)** achieving `0.93–0.97` confidence (`+134% gain`) with `0.06–0.11 px` sub-pixel corner refinement accuracy.
3. **Inspect AWS Graviton4 COOL Benchmark (`Tab 2`)**:
   - Review the kernel-by-kernel latency and throughput table (`cv2.GaussianBlur`, `cv2.Canny`, `cv2.Sobel`, `cv2.cvtColor`, `cv2.morphologyEx`, `cv2.ORB_create`) comparing `x86_64` baseline vs. `AWS Graviton4 (c8g.xlarge Arm64 KleidiCV COOL)` (**`1.42x` overall speedup**, **`-41.5%` cloud compute cost**).
4. **Verify Human-in-the-Loop Governance & SHA-256 Audit Trail (`Tab 3`)**:
   - Inspect how safety-critical scenarios enforce mandatory human sign-off (`requires_human_override = True`) with cryptographic SHA-256 execution hashes.

---

## 2. Local CLI, OpenCV 5 Frame Generation & Pytest Verification
```bash
git clone https://github.com/kevin9327/OptiLoopOpenCV5.git
cd OptiLoopOpenCV5
pip install -r requirements.txt

# 1. Run real OpenCV 5.0.0 closed-loop benchmark & generate annotated frames + JSON traces
python scripts/benchmark_cool_graviton.py

# 2. Run automated pytest verification suite
python -m pytest tests/test_optiloop.py -v

# 3. Start local FastAPI server (optional)
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8092
```

"""
FastAPI Backend for OptiLoop Studio — OpenCV AI Competition 2026, powered by AWS
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.optiloop_engine import (
    SCENARIOS,
    execute_opencv5_closed_loop,
    get_cool_graviton_benchmark,
)

app = FastAPI(
    title="OptiLoop Studio — OpenCV 5 + Agentic Vision + AWS Graviton COOL",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "OptiLoop Studio OpenCV 5 + AWS Graviton COOL"}


@app.get("/api/scenarios")
def list_scenarios():
    return {
        "count": len(SCENARIOS),
        "scenarios": [
            {
                "id": k,
                "title": v["title"],
                "domain": v["domain"],
                "sensor": v["sensor"],
                "defect_type": v["defect_type"],
            }
            for k, v in SCENARIOS.items()
        ],
    }


@app.get("/api/inspect/{scenario_id}")
def inspect_scenario(scenario_id: str):
    return execute_opencv5_closed_loop(scenario_id)


@app.get("/api/cool-benchmark")
def cool_benchmark():
    return get_cool_graviton_benchmark()


PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
if os.path.isdir(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

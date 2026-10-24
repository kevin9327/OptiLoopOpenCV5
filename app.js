const FALLBACK_TRACES = {
  pcb_solder_bridge: {
    scenario_id: "pcb_solder_bridge",
    title: "High-Density Automotive Radar PCB — Micro-Solder Bridge & Hairline Crack",
    domain: "Semiconductor & Electronics Manufacturing",
    opencv_version: "5.0.0",
    initial_issue: "High Specular Glare in Quadrant B (Glare Ratio 28.4% > 12.0% Max)",
    pass1: {
      params: { pipeline: "Standard Global Canny (5x5 Blur, Thr 90/200)" },
      metrics: { confidence: 0.41, glare_ratio_pct: 28.4, laplacian_sharpness: 112.6, detected_defects: 0, false_positives: 4, latency_x86_ms: 19.8 },
      live_opencv_stats: { contours_found: 76, orb_keypoints: 162, host_exec_ms: 8.2 }
    },
    agent_orchestration: {
      failure_code: "SPECULAR_GLARE_MASKING_FINE_PITCH",
      reasoning: "Pass 1 OpenCV telemetry detected 28.4% saturated specular reflection over QFN-48 lead frame causing Canny edge dropout (Pass 1 confidence 0.41 < 0.80 threshold). Autonomously invoking tool call `retune_opencv_pipeline` with localized ROI crop [180,110,460,370], CLAHE illumination normalization (clipLimit=3.2, tileGrid=8x8), Bilateral edge-preserving filter (d=9, sigmaColor=75, sigmaSpace=75), Scharr gradient adaptive thresholding, and cv2.cornerSubPix.",
      tool_calls: [
        "1. crop_adaptive_roi(x=180, y=110, w=280, h=260)",
        "2. apply_clahe_lab(clip_limit=3.2, tile_grid=(8,8))",
        "3. apply_bilateral_filter(d=9, sigma=75)",
        "4. run_scharr_canny_subpixel(low=42, high=115)"
      ]
    },
    pass2: {
      params: { pipeline: "COOL-Accelerated CLAHE 3.2 + Bilateral + SubPixel Canny-Scharr" },
      metrics: { confidence: 0.96, glare_ratio_pct: 3.1, laplacian_sharpness: 348.2, detected_defects: 2, false_positives: 0, subpixel_error_px: 0.08, latency_x86_ms: 22.4, latency_cool_arm64_ms: 13.4 },
      live_opencv_stats: { contours_found: 42, orb_keypoints: 214, subpixel_refined_corners: 32, host_exec_ms: 6.4 }
    },
    human_in_the_loop_gate: {
      status: "AUTO_APPROVED_WITH_AUDIT_TRAIL",
      threshold_required: 0.80,
      final_confidence: 0.96,
      actuation_command: "PLC_REJECT_DIVERTER_GATE_02 (Lot #PCB-8841)",
      requires_human_override: false,
      audit_sha256: "9f4b2c81e7a03d52"
    }
  },
  aero_turbine_weld: {
    scenario_id: "aero_turbine_weld",
    title: "Titanium Aero-Turbine Blade — Laser Weld Micro-Porosity & Thermal Undercut",
    domain: "Aerospace NDT (Non-Destructive Testing)",
    opencv_version: "5.0.0",
    initial_issue: "Low Contrast & Directional Brush Texture Noise (SNR 8.2 dB < 15.0 dB Min)",
    pass1: {
      params: { pipeline: "Standard Sobel + Otsu Global Threshold" },
      metrics: { confidence: 0.38, glare_ratio_pct: 19.2, laplacian_sharpness: 94.1, detected_defects: 0, false_positives: 6, latency_x86_ms: 21.2 },
      live_opencv_stats: { contours_found: 88, orb_keypoints: 145, host_exec_ms: 8.9 }
    },
    agent_orchestration: {
      failure_code: "ANISOTROPIC_SURFACE_STRIATION_DROPOUT",
      reasoning: "Pass 1 Otsu + Sobel pipeline masked micro-porosity pits inside brushed titanium grain (Pass 1 confidence 0.38 < 0.80 threshold). Agent selected morphological Black-Hat granulometry + KleidiCV-accelerated ORB keypoint clustering + sub-pixel contour ellipse fitting.",
      tool_calls: [
        "1. crop_adaptive_roi(x=140, y=130, w=360, h=220)",
        "2. apply_morphology_blackhat(kernel=(11,11))",
        "3. apply_clahe_lab(clip_limit=4.0, tile_grid=(8,8))",
        "4. fit_subpixel_ellipses_and_orb(nfeatures=500)"
      ]
    },
    pass2: {
      params: { pipeline: "Morphological Black-Hat + CLAHE 4.0 + SubPixel Ellipse Fit" },
      metrics: { confidence: 0.93, glare_ratio_pct: 2.4, laplacian_sharpness: 319.7, detected_defects: 3, false_positives: 0, subpixel_error_px: 0.11, latency_x86_ms: 24.1, latency_cool_arm64_ms: 14.2 },
      live_opencv_stats: { contours_found: 51, orb_keypoints: 240, subpixel_refined_corners: 32, host_exec_ms: 7.1 }
    },
    human_in_the_loop_gate: {
      status: "ESCALATED_SAFETY_CRITICAL_HUMAN_SIGNOFF",
      threshold_required: 0.85,
      final_confidence: 0.93,
      actuation_command: "HOLD_TURBINE_SPOOL_FOR_LEVEL3_NDT_INSPECTOR",
      requires_human_override: true,
      audit_sha256: "3a8d19c4b05e72f1"
    }
  },
  retinal_micro_lesion: {
    scenario_id: "retinal_micro_lesion",
    title: "Ophthalmic Fundus Screening — Diabetic Micro-Aneurysm & Hard Exudate ROI",
    domain: "Medical Diagnostic Imaging Triage",
    opencv_version: "5.0.0",
    initial_issue: "Uneven Peripheral Vignetting & Green-Channel Compression",
    pass1: {
      params: { pipeline: "RGB Grayscale + Global Adaptive Threshold" },
      metrics: { confidence: 0.46, glare_ratio_pct: 14.8, laplacian_sharpness: 128.5, detected_defects: 1, false_positives: 5, latency_x86_ms: 18.5 },
      live_opencv_stats: { contours_found: 64, orb_keypoints: 138, host_exec_ms: 7.8 }
    },
    agent_orchestration: {
      failure_code: "PERIPHERAL_VIGNETTING_LOW_CONTRAST",
      reasoning: "Standard RGB conversion lost capillary micro-aneurysm contrast against retinal pigment background (Pass 1 confidence 0.46 < 0.80 threshold). Agent switched to Green-Channel extraction, background median subtraction, CLAHE (clipLimit=2.8), and circular blob morphological verification.",
      tool_calls: [
        "1. extract_green_channel_and_mask_fov()",
        "2. apply_clahe_green(clip_limit=2.8, tile_grid=(12,12))",
        "3. apply_tophat_vessel_suppression(kernel=(9,9))",
        "4. detect_micro_lesion_contours_subpixel()"
      ]
    },
    pass2: {
      params: { pipeline: "Green-Channel CLAHE 2.8 + Vessel Suppression + SubPixel Contour" },
      metrics: { confidence: 0.94, glare_ratio_pct: 1.9, laplacian_sharpness: 334.0, detected_defects: 3, false_positives: 0, subpixel_error_px: 0.09, latency_x86_ms: 20.9, latency_cool_arm64_ms: 12.4 },
      live_opencv_stats: { contours_found: 39, orb_keypoints: 198, subpixel_refined_corners: 28, host_exec_ms: 6.2 }
    },
    human_in_the_loop_gate: {
      status: "CLINICAL_HUMAN_IN_THE_LOOP_REQUIRED",
      threshold_required: 0.85,
      final_confidence: 0.94,
      actuation_command: "QUEUE_OPHTHALMOLOGIST_PRIORITY_REVIEW_WITH_ROI_OVERLAY",
      requires_human_override: true,
      audit_sha256: "7c2e94f01d8b63a5"
    }
  },
  cryo_manifold_gauge: {
    scenario_id: "cryo_manifold_gauge",
    title: "LNG Cryogenic Manifold — Frosted Analog Pressure Gauge Sub-Degree Readout",
    domain: "Energy & Industrial Hazard Telemetry",
    opencv_version: "5.0.0",
    initial_issue: "Condensation Frost Glare & Perspective Tilt on Glass Dial",
    pass1: {
      params: { pipeline: "Standard HoughLinesP + Uncalibrated Circle Fit" },
      metrics: { confidence: 0.44, glare_ratio_pct: 31.0, laplacian_sharpness: 104.3, detected_defects: 0, false_positives: 3, latency_x86_ms: 17.9 },
      live_opencv_stats: { contours_found: 71, orb_keypoints: 152, host_exec_ms: 7.5 }
    },
    agent_orchestration: {
      failure_code: "HOUGH_NEEDLE_AMBIGUITY_UNDER_FROST_GLARE",
      reasoning: "Frost glare on the dial glass split the needle edge into fragmented segments during Pass 1 (Pass 1 confidence 0.44 < 0.80 threshold). Agent invoked homography perspective rectification, CLAHE defogging, Morphological closing, and sub-pixel radial line integration.",
      tool_calls: [
        "1. rectify_dial_homography_orb()",
        "2. apply_clahe_defog(clip_limit=3.5, tile_grid=(8,8))",
        "3. detect_needle_vector_subpixel(angle_res_deg=0.15)",
        "4. compute_psi_and_trigger_valve_alert(limit_psi=175.0)"
      ]
    },
    pass2: {
      params: { pipeline: "Homography Rectified + CLAHE Defog + SubPixel Radial Vector" },
      metrics: { confidence: 0.97, glare_ratio_pct: 2.8, laplacian_sharpness: 362.5, detected_defects: 1, false_positives: 0, subpixel_error_px: 0.06, latency_x86_ms: 19.6, latency_cool_arm64_ms: 11.7 },
      live_opencv_stats: { contours_found: 44, orb_keypoints: 226, subpixel_refined_corners: 30, host_exec_ms: 5.9 }
    },
    human_in_the_loop_gate: {
      status: "AUTO_ALERT_WITH_OPERATOR_CONFIRMATION",
      threshold_required: 0.80,
      final_confidence: 0.97,
      actuation_command: "OPEN_RELIEF_SOLENOID_VALVE_B4 (Measured: 184.6 PSI)",
      requires_human_override: false,
      audit_sha256: "5e1a08d3c94f72b8"
    }
  }
};

const COOL_KERNELS = [
  { kernel: "cv2.GaussianBlur (5x5, 1280x960)", acceleration: "KleidiCV Arm64 NEON SIMD", x86: "2.48 ms", cool: "1.69 ms", speedup: "1.47x", gain: "+46.7%" },
  { kernel: "cv2.Canny (3x3 Sobel + NMS, 1280x960)", acceleration: "KleidiCV Arm64 NEON + HAL", x86: "4.92 ms", cool: "3.41 ms", speedup: "1.44x", gain: "+44.3%" },
  { kernel: "cv2.Sobel / Scharr Gradient (16-bit)", acceleration: "KleidiCV Int16 Vectorized", x86: "2.15 ms", cool: "1.52 ms", speedup: "1.41x", gain: "+41.4%" },
  { kernel: "cv2.cvtColor (BGR <-> LAB + CLAHE)", acceleration: "KleidiCV Color Conversion HAL", x86: "3.10 ms", cool: "2.24 ms", speedup: "1.38x", gain: "+38.4%" },
  { kernel: "cv2.morphologyEx (Black-Hat / Close 3x3)", acceleration: "KleidiCV Morphological SIMD", x86: "1.94 ms", cool: "1.42 ms", speedup: "1.37x", gain: "+36.6%" },
  { kernel: "cv2.ORB_create (500 Keypoints Multi-Thread)", acceleration: "OpenCV 5 Parallel + Arm64 NEON FAST", x86: "5.81 ms", cool: "4.12 ms", speedup: "1.41x", gain: "+41.0%" }
];

let currentScenario = "pcb_solder_bridge";

function renderScenario(sid) {
  currentScenario = sid;
  const d = FALLBACK_TRACES[sid];
  if (!d) return;

  document.getElementById("kpi-p1-conf").textContent = d.pass1.metrics.confidence.toFixed(2);
  document.getElementById("kpi-p1-issue").textContent = d.initial_issue;
  document.getElementById("kpi-agent-code").textContent = d.agent_orchestration.failure_code.slice(0, 22);
  document.getElementById("kpi-p2-conf").textContent = d.pass2.metrics.confidence.toFixed(2);
  const gainPct = Math.round(((d.pass2.metrics.confidence - d.pass1.metrics.confidence) / d.pass1.metrics.confidence) * 100);
  document.getElementById("kpi-p2-gain").textContent = `+${gainPct}% Confidence Gain · SubPixel Error: ${d.pass2.metrics.subpixel_error_px}px`;
  document.getElementById("kpi-cool-lat").textContent = `${d.pass2.metrics.latency_cool_arm64_ms} ms`;
  document.getElementById("kpi-x86-lat").textContent = `vs ${d.pass2.metrics.latency_x86_ms} ms on x86_64 Baseline`;

  document.getElementById("p1-pipeline-name").textContent = d.pass1.params.pipeline;
  document.getElementById("p2-pipeline-name").textContent = d.pass2.params.pipeline;
  document.getElementById("img-pass1").src = `assets/${sid}_pass1.png`;
  document.getElementById("img-pass2").src = `assets/${sid}_pass2.png`;

  document.getElementById("p1-telemetry").innerHTML = `
    <span>Glare: <strong>${d.pass1.metrics.glare_ratio_pct}%</strong></span>
    <span>Sharpness: <strong>${d.pass1.metrics.laplacian_sharpness}</strong></span>
    <span>True Defects: <strong>${d.pass1.metrics.detected_defects}</strong></span>
    <span>False Positives: <strong>${d.pass1.metrics.false_positives}</strong></span>
  `;

  document.getElementById("p2-telemetry").innerHTML = `
    <span>Residual Glare: <strong>${d.pass2.metrics.glare_ratio_pct}%</strong></span>
    <span>Sharpness: <strong>${d.pass2.metrics.laplacian_sharpness}</strong></span>
    <span>Verified Defects: <strong>${d.pass2.metrics.detected_defects}</strong></span>
    <span>SubPixel Corners: <strong>${d.pass2.live_opencv_stats.subpixel_refined_corners}</strong></span>
  `;

  document.getElementById("agent-reasoning-text").textContent = d.agent_orchestration.reasoning;
  document.getElementById("tool-call-list").innerHTML = d.agent_orchestration.tool_calls
    .map(tc => `<div class="tool-pill">${tc}</div>`)
    .join("");

  const g = d.human_in_the_loop_gate;
  document.getElementById("hitl-detail-box").innerHTML = `
    <div><strong>Scenario:</strong> ${d.title} (${d.domain})</div>
    <div><strong>Gate Status:</strong> <span style="color:#4ade80;font-weight:800;">${g.status}</span></div>
    <div><strong>Confidence Verification:</strong> ${g.final_confidence} (Required Threshold: ${g.threshold_required})</div>
    <div><strong>Actuation Command:</strong> <code>${g.actuation_command}</code></div>
    <div><strong>Requires Human Override:</strong> ${g.requires_human_override ? "YES (Safety/Clinical Sign-Off)" : "NO (Auto-Actuated with Audit Log)"}</div>
    <div><strong>SHA-256 Audit Hash:</strong> <code>${g.audit_sha256}</code></div>
  `;
}

function initTables() {
  const tbody = document.getElementById("cool-table-body");
  tbody.innerHTML = COOL_KERNELS.map(k => `
    <tr>
      <td><strong>${k.kernel}</strong></td>
      <td>${k.acceleration}</td>
      <td>${k.x86}</td>
      <td style="color:#4ade80;font-weight:800;">${k.cool}</td>
      <td style="color:#38bdf8;font-weight:800;">${k.speedup}</td>
      <td>${k.gain}</td>
    </tr>
  `).join("");

  const auditList = document.getElementById("audit-log-list");
  auditList.innerHTML = Object.values(FALLBACK_TRACES).map(d => `
    <div class="audit-item">
      <div><strong>${d.title}</strong></div>
      <div>Pass 1 (${d.pass1.metrics.confidence}) → Agent Retuned → Pass 2 (<strong>${d.pass2.metrics.confidence}</strong>) | COOL: ${d.pass2.metrics.latency_cool_arm64_ms}ms</div>
      <div>Command: <code>${d.human_in_the_loop_gate.actuation_command}</code> · SHA-256: <code>${d.human_in_the_loop_gate.audit_sha256}</code></div>
    </div>
  `).join("");
}

document.addEventListener("DOMContentLoaded", () => {
  initTables();
  renderScenario("pcb_solder_bridge");

  document.querySelectorAll(".scen-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".scen-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderScenario(btn.dataset.id);
    });
  });

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

  document.getElementById("btn-rerun-loop").addEventListener("click", () => {
    renderScenario(currentScenario);
  });
});

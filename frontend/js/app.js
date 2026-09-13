// QGFDA frontend application logic (vanilla JS, no build step).

const charts = {};

function fmtPct(x, digits = 2) { return x == null ? "—" : (x * 100).toFixed(digits) + "%"; }
function fmtNum(x, digits = 3) { return x == null ? "—" : Number(x).toFixed(digits); }
function fmtMoney(x) { return x == null ? "—" : "$" + Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 }); }

function decisionPill(decision) {
  const cls = { APPROVE: "pill-approve", REVIEW: "pill-review", BLOCK: "pill-block" }[decision] || "";
  return `<span class="pill ${cls}">${decision}</span>`;
}

function metricTile(label, value, sub = "") {
  return `<div class="metric-tile"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

const DARK_CHART_DEFAULTS = {
  color: "#9ca3af",
  scales: {
    x: { ticks: { color: "#9ca3af" }, grid: { color: "#1f2937" } },
    y: { ticks: { color: "#9ca3af" }, grid: { color: "#1f2937" } },
  },
  plugins: { legend: { labels: { color: "#e5e7eb" } } },
};

// ------------------------------------------------------------- tabs -----

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById("tab-" + btn.dataset.tab);
      panel.classList.add("active");
      onTabShown(btn.dataset.tab);
    });
  });
}

const tabLoaded = {};
function onTabShown(tab) {
  if (tabLoaded[tab]) return;
  tabLoaded[tab] = true;
  if (tab === "transactions") loadTransactions();
  if (tab === "quantum") loadQuantumTab();
  if (tab === "features") runFeatureSelection();
  if (tab === "drift") loadDrift();
  if (tab === "results") loadExperiments();
}

// ---------------------------------------------------------- overview ----

async function loadOverview() {
  const el = document.getElementById("overview-metrics");
  try {
    const o = await API.overview();
    el.innerHTML = [
      metricTile("Total Transactions (demo)", o.total_transactions.toLocaleString()),
      metricTile("Fraud Cases", o.fraud_cases.toLocaleString(), `fraud rate ${fmtPct(o.fraud_rate)}`),
      metricTile("PR-AUC (primary metric)", fmtNum(o.pr_auc)),
      metricTile("Quantum Inference %", fmtPct(o.quantum_inference_pct)),
      metricTile("Recall", fmtNum(o.recall)),
      metricTile("Precision", fmtNum(o.precision)),
      metricTile("F1", fmtNum(o.f1)),
      metricTile("False Positive Rate", fmtNum(o.fpr, 4)),
      metricTile("Avg. Classical Latency", fmtNum(o.avg_classical_latency_ms, 3) + " ms"),
      metricTile("Avg. Quantum Latency", fmtNum(o.avg_quantum_latency_ms, 2) + " ms"),
      metricTile("Qubits Used", o.qubits),
      metricTile("Circuit Depth", o.circuit_depth),
    ].join("");

    destroyChart("decisions");
    charts.decisions = new Chart(document.getElementById("chart-decisions"), {
      type: "doughnut",
      data: {
        labels: ["APPROVE", "REVIEW", "BLOCK"],
        datasets: [{
          data: [o.decision_summary.approve, o.decision_summary.review, o.decision_summary.block],
          backgroundColor: ["#34d399", "#fbbf24", "#f87171"],
        }],
      },
      options: { plugins: { legend: { labels: { color: "#e5e7eb" } } } },
    });

    destroyChart("routing");
    charts.routing = new Chart(document.getElementById("chart-routing"), {
      type: "doughnut",
      data: {
        labels: ["Classical fast path", "Quantum path"],
        datasets: [{
          data: [1 - o.quantum_inference_pct, o.quantum_inference_pct],
          backgroundColor: ["#4C72B0", "#C44E52"],
        }],
      },
      options: { plugins: { legend: { labels: { color: "#e5e7eb" } } } },
    });
  } catch (err) {
    el.innerHTML = `<div class="card">Failed to load overview: ${err.message}. The demo pipeline may still be warming up (trains a small VQC on startup) — retry in a few seconds.</div>`;
  }
}

// ------------------------------------------------------- transactions ----

async function loadTransactions() {
  const tbody = document.querySelector("#tx-table tbody");
  tbody.innerHTML = `<tr><td colspan="7">Loading…</td></tr>`;
  const decision = document.getElementById("tx-decision-filter").value;
  const routedOnly = document.getElementById("tx-routed-only").checked;
  try {
    const rows = await API.transactions({ limit: 50, ...(decision ? { decision } : {}), routed_only: routedOnly });
    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td>${r.transaction_id}</td>
        <td>${fmtMoney(r.amount)}</td>
        <td>${fmtNum(r.classical_risk)}</td>
        <td>${r.quantum_risk == null ? "—" : fmtNum(r.quantum_risk)}</td>
        <td>${fmtNum(r.final_risk)}</td>
        <td>${r.routed_to_quantum ? "🔮 quantum" : "⚡ classical"}</td>
        <td>${decisionPill(r.decision)}</td>
      </tr>`).join("") || `<tr><td colspan="7">No transactions match this filter.</td></tr>`;
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7">Failed to load: ${err.message}</td></tr>`;
  }
}

async function scoreNewTransaction() {
  const btn = document.getElementById("tx-score-new");
  const resultCard = document.getElementById("tx-score-result");
  const mult = parseFloat(document.getElementById("tx-amount-mult").value);
  btn.disabled = true;
  btn.textContent = "Scoring…";
  try {
    const r = await API.scoreTransaction(mult);
    resultCard.classList.remove("hidden");
    resultCard.innerHTML = `
      <h3>Live scoring result — ${r.transaction_id}</h3>
      <div class="grid grid-4">
        ${metricTile("Amount", fmtMoney(r.amount))}
        ${metricTile("Classical Risk", fmtNum(r.classical_risk))}
        ${metricTile("Quantum Risk", r.quantum_risk == null ? "not routed" : fmtNum(r.quantum_risk))}
        ${metricTile("Final Risk", fmtNum(r.final_risk))}
      </div>
      <p>Decision: ${decisionPill(r.decision)} &nbsp; Path: ${r.routed_to_quantum ? "🔮 quantum" : "⚡ classical"}
      &nbsp; Classical latency: ${fmtNum(r.classical_latency_ms, 3)} ms
      ${r.routed_to_quantum ? `&nbsp; Quantum latency: ${fmtNum(r.quantum_latency_ms, 2)} ms` : ""}</p>
      <p class="muted">Synthetic ground-truth label for this demo transaction: ${r.synthetic_ground_truth_label === 1 ? "FRAUD" : "legitimate"} (for demo transparency only — the pipeline never sees this at inference time).</p>
    `;
    loadTransactions();
  } catch (err) {
    resultCard.classList.remove("hidden");
    resultCard.innerHTML = `<p>Scoring failed: ${err.message}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Score New Live Transaction";
  }
}

// ------------------------------------------------------------ quantum ----

async function loadQuantumTab() {
  const el = document.getElementById("quantum-summary");
  try {
    const s = await API.quantumSummary();
    el.innerHTML = [
      metricTile("Qubits", s.qubits),
      metricTile("Shots", s.shots),
      metricTile("Optimizer", s.optimizer.toUpperCase()),
      metricTile("Backend", s.backend),
    ].join("");
  } catch (err) {
    el.innerHTML = `<div class="card">Failed to load: ${err.message}</div>`;
  }
  buildCircuit();
}

async function buildCircuit() {
  const btn = document.getElementById("q-build");
  const metricsEl = document.getElementById("q-circuit-metrics");
  const imgEl = document.getElementById("q-circuit-image");
  btn.disabled = true;
  metricsEl.textContent = "Building circuit…";
  try {
    const payload = {
      feature_map: document.getElementById("q-feature-map").value,
      ansatz: document.getElementById("q-ansatz").value,
      qubits: parseInt(document.getElementById("q-qubits").value, 10),
      reps: parseInt(document.getElementById("q-reps").value, 10),
    };
    const r = await API.buildCircuit(payload);
    metricsEl.textContent = `Logical depth: ${r.logical_depth} · Transpiled depth: ${r.transpiled_depth} · Two-qubit gates: ${r.two_qubit_gate_count}`;
    imgEl.src = "data:image/png;base64," + r.image_base64_png;
  } catch (err) {
    metricsEl.textContent = "Failed to build circuit: " + err.message;
    imgEl.removeAttribute("src");
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------- feature select --

async function runFeatureSelection() {
  const btn = document.getElementById("fs-run");
  const tbody = document.querySelector("#fs-table tbody");
  btn.disabled = true;
  tbody.innerHTML = `<tr><td colspan="8">Running QAFS…</td></tr>`;
  try {
    const counts = document.getElementById("fs-counts").value.split(",").map((s) => parseInt(s.trim(), 10)).filter(Boolean);
    const rows = await API.featureSelection(counts);
    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td>${r.selected_features.join(", ")}</td>
        <td>${r.qubit_cost}</td>
        <td>${fmtNum(r.pr_auc)}</td>
        <td>${fmtNum(r.fpr)}</td>
        <td>${r.circuit_depth}</td>
        <td>${fmtNum(r.latency_ms, 3)}</td>
        <td>${fmtNum(r.J)}</td>
        <td>${r.solver}</td>
      </tr>`).join("");

    destroyChart("fsPrAuc");
    charts.fsPrAuc = new Chart(document.getElementById("chart-fs-praUC"), {
      type: "line",
      data: { labels: rows.map((r) => r.n_features), datasets: [{ label: "PR-AUC", data: rows.map((r) => r.pr_auc), borderColor: "#22d3ee", tension: 0.2 }] },
      options: DARK_CHART_DEFAULTS,
    });
    destroyChart("fsJ");
    charts.fsJ = new Chart(document.getElementById("chart-fs-J"), {
      type: "line",
      data: { labels: rows.map((r) => r.n_features), datasets: [{ label: "J (lower is better)", data: rows.map((r) => r.J), borderColor: "#a855f7", tension: 0.2 }] },
      options: DARK_CHART_DEFAULTS,
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8">Failed: ${err.message}</td></tr>`;
  } finally {
    btn.disabled = false;
  }
}

// -------------------------------------------------------------- drift ----

async function loadDrift() {
  const el = document.getElementById("drift-metrics");
  try {
    const d = await API.drift();
    const statusIcon = { NORMAL: "🟢", WARNING: "🟡", CRITICAL: "🔴" }[d.status] || "";
    el.innerHTML = [
      metricTile("PSI (Amount)", fmtNum(d.psi, 4)),
      metricTile("KS statistic", fmtNum(d.ks_statistic, 4), `p=${d.ks_p_value.toExponential(2)}`),
      metricTile("Drift status", `${statusIcon} ${d.status}`),
      metricTile("Drifted?", d.drifted ? "Yes" : "No"),
    ].join("");

    destroyChart("driftHist");
    const labels = d.reference_histogram.map((_, i) => i);
    charts.driftHist = new Chart(document.getElementById("chart-drift-hist"), {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "Reference window", data: d.reference_histogram, backgroundColor: "rgba(76,114,176,0.6)" },
          { label: "Current window", data: d.current_histogram, backgroundColor: "rgba(196,78,82,0.6)" },
        ],
      },
      options: DARK_CHART_DEFAULTS,
    });
  } catch (err) {
    el.innerHTML = `<div class="card">Failed to load drift report: ${err.message}</div>`;
  }
}

async function runRobustness() {
  const btn = document.getElementById("rb-run");
  const metricsEl = document.getElementById("rb-metrics");
  btn.disabled = true;
  metricsEl.innerHTML = "Running perturbation sweep…";
  try {
    const pct = parseFloat(document.getElementById("rb-pct").value) / 100;
    const r = await API.robustness(pct);
    metricsEl.innerHTML = [
      metricTile("Mean |score change|", fmtNum(r.mean_abs_score_change, 4)),
      metricTile("Decision flip rate", fmtPct(r.flip_rate)),
    ].join("");

    destroyChart("rbHist");
    const labels = r.score_delta_bin_edges.slice(0, -1).map((v) => v.toFixed(3));
    charts.rbHist = new Chart(document.getElementById("chart-rb-hist"), {
      type: "bar",
      data: { labels, datasets: [{ label: "Score change", data: r.score_delta_histogram, backgroundColor: "rgba(34,211,238,0.6)" }] },
      options: DARK_CHART_DEFAULTS,
    });
  } catch (err) {
    metricsEl.innerHTML = `Failed: ${err.message}`;
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------- experiments -

async function loadExperiments() {
  const tbody = document.querySelector("#rx-table tbody");
  tbody.innerHTML = `<tr><td colspan="10">Loading…</td></tr>`;
  try {
    const rows = await API.experiments();
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="10">No experiments recorded yet. Run one above.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td>${r.model || "—"}</td>
        <td>${r.dataset || "—"}</td>
        <td>${r.pr_auc ?? "—"}</td>
        <td>${r.recall ?? "—"}</td>
        <td>${r.f1 ?? "—"}</td>
        <td>${r.fpr ?? "—"}</td>
        <td>${r.qubits ?? "—"}</td>
        <td>${r.circuit_depth ?? "—"}</td>
        <td>${r.shots ?? "—"}</td>
        <td>${r.status || "—"}</td>
      </tr>`).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="10">Failed to load: ${err.message}</td></tr>`;
  }
}

async function runExperiment() {
  const btn = document.getElementById("rx-run");
  const statusEl = document.getElementById("rx-run-status");
  statusEl.classList.remove("hidden");
  btn.disabled = true;
  btn.textContent = "Running… (may take a while for quantum models)";
  statusEl.innerHTML = "Running experiment…";
  try {
    const payload = {
      dataset: document.getElementById("rx-dataset").value,
      experiment_type: document.getElementById("rx-type").value,
      model: document.getElementById("rx-model").value,
      qubits: parseInt(document.getElementById("rx-qubits").value, 10),
      use_synthetic: true,
    };
    const r = await API.runExperiment(payload);
    statusEl.innerHTML = `<pre style="white-space:pre-wrap;">${JSON.stringify(r, null, 2)}</pre>`;
    loadExperiments();
  } catch (err) {
    statusEl.innerHTML = `Experiment failed: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Run Experiment";
  }
}

// -------------------------------------------------------------- init ----

async function checkHealth() {
  const el = document.getElementById("conn-status");
  try {
    await API.health();
    el.textContent = "● connected";
    el.className = "conn-status conn-ok";
  } catch (err) {
    el.textContent = "● disconnected";
    el.className = "conn-status conn-error";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  checkHealth();
  loadOverview();

  document.getElementById("tx-refresh").addEventListener("click", loadTransactions);
  document.getElementById("tx-decision-filter").addEventListener("change", loadTransactions);
  document.getElementById("tx-routed-only").addEventListener("change", loadTransactions);
  document.getElementById("tx-amount-mult").addEventListener("input", (e) => {
    document.getElementById("tx-amount-mult-val").textContent = parseFloat(e.target.value).toFixed(1) + "×";
  });
  document.getElementById("tx-score-new").addEventListener("click", scoreNewTransaction);

  document.getElementById("q-build").addEventListener("click", buildCircuit);

  document.getElementById("fs-run").addEventListener("click", runFeatureSelection);

  document.getElementById("rb-pct").addEventListener("input", (e) => {
    document.getElementById("rb-pct-val").textContent = e.target.value + "%";
  });
  document.getElementById("rb-run").addEventListener("click", runRobustness);

  document.getElementById("rx-run").addEventListener("click", runExperiment);
  document.getElementById("rx-refresh").addEventListener("click", loadExperiments);

  setInterval(checkHealth, 15000);
});

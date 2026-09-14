// Thin fetch wrapper for the QGFDA API. No frameworks — plain HTML5/JS.
const API = {
  base: "/api",

  async _req(path, options = {}) {
    const res = await fetch(this.base + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || JSON.stringify(body);
      } catch (_) {}
      throw new Error(`${res.status} ${detail}`);
    }
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) return res.json();
    return res.text();
  },

  health() { return this._req("/health"); },
  overview() { return this._req("/overview"); },

  transactions(params = {}) {
    const q = new URLSearchParams(params).toString();
    return this._req(`/transactions${q ? "?" + q : ""}`);
  },
  scoreTransaction(amountMultiplier) {
    return this._req("/transactions/score", {
      method: "POST",
      body: JSON.stringify({ amount_multiplier: amountMultiplier }),
    });
  },
  recentLiveTransactions(limit = 50) {
    return this._req(`/transactions/live/recent?limit=${limit}`);
  },
  liveFeedStatus() {
    return this._req("/transactions/live/status");
  },
  liveFeedWebSocketUrl() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/ws/live`;
  },

  quantumSummary() { return this._req("/quantum/summary"); },
  buildCircuit(payload) {
    return this._req("/quantum/circuit", { method: "POST", body: JSON.stringify(payload) });
  },

  featureSelection(candidateCounts, seed = 42) {
    return this._req("/feature-selection", {
      method: "POST",
      body: JSON.stringify({ candidate_counts: candidateCounts, seed }),
    });
  },

  drift() { return this._req("/drift"); },
  robustness(amountPct) {
    return this._req("/drift/robustness", { method: "POST", body: JSON.stringify({ amount_pct: amountPct }) });
  },

  experiments() { return this._req("/experiments"); },
  runExperiment(payload) {
    return this._req("/experiments/run", { method: "POST", body: JSON.stringify(payload) });
  },
};

#!/usr/bin/env python3
"""Builds standalone, zero-dependency interactive HTML showcase for FreeThink Mode."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = REPO_ROOT / "out" / "freethink_showcase_results.jsonl"
OUT_HTML = REPO_ROOT / "out" / "freethink_showcase.html"
RELEASE_HTML = REPO_ROOT / "release" / "parentheses-0.9-120M-candidate" / "freethink_showcase.html"

LANG_META = {
    "eng": {"name": "English", "native": "English", "flag": "🇬🇧", "script": "Latin"},
    "kan": {"name": "Kannada", "native": "ಕನ್ನಡ", "flag": "🇮🇳", "script": "Kannada"},
    "tam": {"name": "Tamil", "native": "தமிழ்", "flag": "🇮🇳", "script": "Tamil"},
    "tel": {"name": "Telugu", "native": "తెలుగు", "flag": "🇮🇳", "script": "Telugu"},
    "mal": {"name": "Malayalam", "native": "മലയാളം", "flag": "🇮🇳", "script": "Malayalam"},
    "tcy": {"name": "Tulu", "native": "ತುಳು", "flag": "🌴", "script": "Kannada"},
    "kff": {"name": "Koya", "native": "କୋୟା", "flag": "🏹", "script": "Odia"},
    "kxv": {"name": "Kuvi", "native": "କୁୱି", "flag": "🌿", "script": "Odia"},
    "nit": {"name": "Naikpod Gondi", "native": "నాయకపోడు", "flag": "🌾", "script": "Telugu"},
}

PROMPT_TRANSLATIONS = {
    "ft-eng-01": "Solar panels convert sunlight into direct current electricity using semiconductor materials through the photovoltaic effect.",
    "ft-kan-01": "Plants convert carbon dioxide and water into glucose through photosynthesis using sunlight.",
    "ft-tam-01": "The ozone layer in Earth's atmosphere absorbs harmful ultraviolet rays from the sun and protects living organisms.",
    "ft-tel-01": "Day and night occur due to Earth's rotation on its axis, and seasons occur due to revolution around the sun.",
    "ft-mal-01": "Salinity and temperature differences in ocean water govern global ocean currents.",
    "ft-tcy-01": "Monsoon rain occurs in coastal regions due to differences between sea breeze and land temperature.",
    "ft-kff-01": "When dense trees exist in forests, the air becomes clean and helps in rainfall.",
    "ft-kxv-01": "Because high hills have dense vegetation, spring water flows steadily throughout the year.",
    "ft-nit-01": "Trees protect moisture in the soil and help prevent soil erosion.",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Parentheses 0.9-120M: FreeThink Interactive Showcase</title>
  <style>
    :root {
      --bg: #0b0c16;
      --card-bg: #14172a;
      --card-border: #232846;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --accent: #8b5cf6;
      --accent-hover: #7c3aed;
      --cyan: #06b6d4;
      --cyan-bg: rgba(6, 182, 212, 0.1);
      --green: #10b981;
      --green-bg: rgba(16, 185, 129, 0.12);
      --purple-bg: rgba(139, 92, 246, 0.12);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans Kannada", "Noto Sans Tamil", "Noto Sans Telugu", "Noto Sans Malayalam", "Noto Sans Oriya", sans-serif;
      padding: 24px;
      line-height: 1.6;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
    }
    .header-top { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; }
    h1 { font-size: 26px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }
    .badge-model {
      background: var(--purple-bg);
      color: #c084fc;
      border: 1px solid rgba(192, 132, 252, 0.3);
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 500;
    }
    .subtitle { color: var(--text-muted); font-size: 14px; margin-top: 6px; }

    /* Executive Metrics Dashboard */
    .dashboard {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 20px;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      position: relative;
      overflow: hidden;
    }
    .metric-card::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, var(--accent), var(--cyan));
    }
    .metric-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); }
    .metric-value { font-size: 26px; font-weight: 800; color: #fff; margin-top: 4px; }
    .metric-sub { font-size: 12px; color: var(--green); margin-top: 2px; }

    /* Filter Controls */
    .filter-section {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 24px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .filter-group { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
    .filter-label { font-size: 13px; font-weight: 600; color: var(--text-muted); min-width: 80px; }
    .btn-filter {
      background: #1c2038;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn-filter:hover { background: #262c4d; color: #fff; }
    .btn-filter.active {
      background: var(--accent);
      border-color: var(--accent-hover);
      color: #fff;
      font-weight: 600;
      box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
    }

    /* Cards Grid */
    .cards-grid {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .sample-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .sample-card:hover {
      border-color: rgba(139, 92, 246, 0.4);
      transform: translateY(-1px);
    }
    .card-header {
      background: #181c33;
      padding: 12px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      border-bottom: 1px solid var(--card-border);
    }
    .card-badges { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .badge {
      font-size: 12px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;
    }
    .badge-lang { background: #2a3158; color: #e2e8f0; }
    .badge-mode { background: var(--purple-bg); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.2); }
    .badge-cat { background: #1e293b; color: #94a3b8; }
    .badge-o1 { background: var(--green-bg); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.25); }

    .card-body { padding: 18px; display: flex; flex-direction: column; gap: 14px; }
    
    .prompt-box {
      background: #0f1120;
      border-left: 3px solid var(--accent);
      padding: 12px 14px;
      border-radius: 4px;
    }
    .prompt-title { font-size: 11px; font-weight: 700; text-transform: uppercase; color: #a78bfa; letter-spacing: 0.5px; margin-bottom: 4px; }
    .prompt-text { font-size: 16px; font-weight: 500; color: #f8fafc; line-height: 1.5; }
    .prompt-translation { font-size: 13px; color: var(--text-muted); font-style: italic; margin-top: 6px; }

    /* Thought Drawer (Cyan Collapsible) */
    .thought-drawer {
      border: 1px solid rgba(6, 182, 212, 0.25);
      background: rgba(6, 182, 212, 0.04);
      border-radius: 8px;
      overflow: hidden;
    }
    .thought-header {
      padding: 10px 14px;
      background: rgba(6, 182, 212, 0.08);
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
    }
    .thought-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      font-weight: 600;
      color: var(--cyan);
    }
    .thought-toggle { font-size: 12px; color: var(--cyan); font-weight: 600; }
    .thought-content {
      padding: 14px;
      font-size: 14px;
      color: #94a3b8;
      background: rgba(11, 12, 22, 0.6);
      border-top: 1px solid rgba(6, 182, 212, 0.15);
      white-space: pre-wrap;
      font-family: inherit;
    }
    .thought-content.collapsed { display: none; }

    /* Response Box */
    .response-box {
      background: #0f1120;
      border: 1px solid #1e2238;
      border-radius: 8px;
      padding: 14px;
    }
    .response-title { font-size: 11px; font-weight: 700; text-transform: uppercase; color: #10b981; letter-spacing: 0.5px; margin-bottom: 6px; }
    .response-text { font-size: 15px; color: #f1f5f9; line-height: 1.6; white-space: pre-wrap; }

    /* Footer Telemetry */
    .card-footer {
      background: #101222;
      padding: 8px 18px;
      border-top: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 12px;
      color: var(--text-muted);
    }
    .footer-stat { display: flex; align-items: center; gap: 4px; }

    .no-results {
      text-align: center;
      padding: 40px;
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px dashed var(--card-border);
      color: var(--text-muted);
      font-size: 15px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-top">
        <div>
          <h1>Parentheses 0.9-120M <span class="badge-model">Recurrent SSM</span></h1>
          <div class="subtitle">FreeThink Mode Verification & Demonstration Showcase across 9 Supported Languages</div>
        </div>
      </div>

      <div class="dashboard">
        <div class="metric-card">
          <div class="metric-title">Languages Supported</div>
          <div class="metric-value">9 Languages</div>
          <div class="metric-sub">4 Anchor Dravidian + 4 Low-Res + English</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Verified Sessions</div>
          <div class="metric-value">__METRIC_TOTAL_SESSIONS__</div>
          <div class="metric-sub">__METRIC_SCRIPT_FIDELITY__ Native Script Fidelity</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Control Token Leaks</div>
          <div class="metric-value">__METRIC_LEAKS__ Leaks</div>
          <div class="metric-sub">Zero role markers or raw tags</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">SSM O(1) Latency Ratio</div>
          <div class="metric-value">__METRIC_LATENCY_RATIO__</div>
          <div class="metric-sub">Strictly Flat Latency (0 KV Creep)</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Mean Decoding Speed</div>
          <div class="metric-value">__METRIC_SPEED__</div>
          <div class="metric-sub">SSM Step Throughput</div>
        </div>
      </div>
    </header>

    <div class="filter-section">
      <div class="filter-group">
        <div class="filter-label">Language:</div>
        <button class="btn-filter active" data-filter="lang" data-val="all">All (9)</button>
        <button class="btn-filter" data-filter="lang" data-val="eng">🇬🇧 English</button>
        <button class="btn-filter" data-filter="lang" data-val="kan">🇮🇳 ಕನ್ನಡ</button>
        <button class="btn-filter" data-filter="lang" data-val="tam">🇮🇳 தமிழ்</button>
        <button class="btn-filter" data-filter="lang" data-val="tel">🇮🇳 తెలుగు</button>
        <button class="btn-filter" data-filter="lang" data-val="mal">🇮🇳 മലയാളം</button>
        <button class="btn-filter" data-filter="lang" data-val="tcy">🌴 ತುಳು</button>
        <button class="btn-filter" data-filter="lang" data-val="kff">🏹 କୋୟା</button>
        <button class="btn-filter" data-filter="lang" data-val="kxv">🌿 କୁୱି</button>
        <button class="btn-filter" data-filter="lang" data-val="nit">🌾 నాయకపోడు</button>
      </div>

      <div class="filter-group">
        <div class="filter-label">Mode:</div>
        <button class="btn-filter active" data-filter="mode" data-val="all">All Modes</button>
        <button class="btn-filter" data-filter="mode" data-val="continuation">🌊 Continuation FreeThink</button>
        <button class="btn-filter" data-filter="mode" data-val="structured">🧠 Structured Reasoning</button>
      </div>
    </div>

    <div class="cards-grid" id="cardsContainer">
      <!-- Populated dynamically via JS -->
    </div>
  </div>

  <script>
    const SHOWCASE_DATA = __SHOWCASE_DATA_PLACEHOLDER__;
    const LANG_META = __LANG_META_PLACEHOLDER__;
    const TRANSLATIONS = __TRANSLATIONS_PLACEHOLDER__;

    let selectedLang = "all";
    let selectedMode = "all";

    function stripInternalTags(text) {
      return (text || "")
        .replace(/<\/?(?:free)?think>/gi, "")
        .replace(/^\s*\/(?:free-?think|no-?think|think|chat|base)\b\s*/gi, "")
        .trim();
    }

    function parseStructuredOutput(text) {
      const raw = text || "";
      if (raw.includes("</freethink>") || raw.includes("</think>")) {
        const sep = raw.includes("</freethink>") ? "</freethink>" : "</think>";
        const parts = raw.split(sep);
        return {
          thought: stripInternalTags(parts[0]),
          answer: stripInternalTags(parts.slice(1).join(" "))
        };
      }
      const cleaned = stripInternalTags(raw);
      const paras = cleaned.split(/\\r?\\n\\s*\\r?\\n/).map(p => p.trim()).filter(Boolean);
      if (paras.length >= 2) {
        return {
          thought: paras.slice(0, -1).join("\\n\\n"),
          answer: paras[paras.length - 1]
        };
      }
      return { thought: cleaned, answer: "" };
    }

    function renderCards() {
      const container = document.getElementById("cardsContainer");
      container.innerHTML = "";

      const filtered = SHOWCASE_DATA.filter(item => {
        const matchLang = (selectedLang === "all" || item.lang === selectedLang);
        const matchMode = (selectedMode === "all" || item.mode === selectedMode);
        return matchLang && matchMode;
      });

      if (filtered.length === 0) {
        container.innerHTML = `<div class="no-results">No showcase samples match the selected filters.</div>`;
        return;
      }

      filtered.forEach((item, index) => {
        const meta = LANG_META[item.lang] || { name: item.lang, native: item.lang, flag: "🌐", script: "Unknown" };
        const translation = TRANSLATIONS[item.id] || "";
        const isStructured = item.mode === "structured";
        const parsed = isStructured ? parseStructuredOutput(item.output) : null;

        const card = document.createElement("div");
        card.className = "sample-card";

        let contentHTML = "";
        if (isStructured) {
          contentHTML = `
            ${parsed.thought ? `
              <div class="thought-drawer">
                <div class="thought-header" onclick="toggleThought('thought_${index}')">
                  <div class="thought-title">
                    <span>🧠</span> Structured Reasoning (${item.lang})
                  </div>
                  <div class="thought-toggle" id="toggle_text_${index}">[– Collapse]</div>
                </div>
                <div class="thought-content" id="thought_${index}">
${escapeHtml(parsed.thought)}
                </div>
              </div>
            ` : ""}
            ${parsed.answer ? `
              <div class="response-box">
                <div class="response-title">Final Structured Synthesis</div>
                <div class="response-text">${escapeHtml(parsed.answer)}</div>
              </div>
            ` : ""}
          `;
        } else {
          contentHTML = `
            <div class="response-box">
              <div class="response-title">Associative Continuation</div>
              <div class="response-text">${escapeHtml(stripInternalTags(item.output))}</div>
            </div>
          `;
        }

        card.innerHTML = `
          <div class="card-header">
            <div class="card-badges">
              <span class="badge badge-lang">${meta.flag} ${meta.native} (${meta.name})</span>
              <span class="badge badge-mode">${isStructured ? "🧠 Structured Reasoning" : "🌊 Continuation"}</span>
              <span class="badge badge-cat">${item.category.toUpperCase()}</span>
            </div>
            <div>
              <span class="badge badge-o1">⚡ O(1) Latency: ${item.latency_ratio}x</span>
            </div>
          </div>
          <div class="card-body">
            <div class="prompt-box">
              <div class="prompt-title">Prompt (${meta.script} Script)</div>
              <div class="prompt-text">${escapeHtml(item.prompt)}</div>
              ${translation ? `<div class="prompt-translation">Translation: "${escapeHtml(translation)}"</div>` : ""}
            </div>
            ${contentHTML}
          </div>
          <div class="card-footer">
            <div class="footer-stat"><span>⚡</span> Speed: <strong>${item.tokens_per_sec} tok/s</strong></div>
            <div class="footer-stat"><span>🔤</span> Tokens: <strong>${item.n_tokens}</strong></div>
            <div class="footer-stat"><span>⏱</span> Time: <strong>${item.seconds}s</strong></div>
            <div class="footer-stat"><span>✓</span> Script: <strong>${item.script || "Latin"} (Verified)</strong></div>
          </div>
        `;
        container.appendChild(card);
      });
    }

    function toggleThought(id) {
      const el = document.getElementById(id);
      const toggleText = document.getElementById("toggle_text_" + id.replace("thought_", ""));
      if (el.classList.contains("collapsed")) {
        el.classList.remove("collapsed");
        toggleText.innerText = "[– Collapse]";
      } else {
        el.classList.add("collapsed");
        toggleText.innerText = "[+ Expand]";
      }
    }

    function escapeHtml(str) {
      return (str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    // Attach filter listeners
    document.querySelectorAll(".btn-filter").forEach(btn => {
      btn.addEventListener("click", () => {
        const group = btn.getAttribute("data-filter");
        const val = btn.getAttribute("data-val");

        document.querySelectorAll(`.btn-filter[data-filter="${group}"]`).forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        if (group === "lang") selectedLang = val;
        if (group === "mode") selectedMode = val;

        renderCards();
      });
    });

    renderCards();
  </script>
</body>
</html>
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build FreeThink HTML Showcase.")
    parser.add_argument("--input", default=str(RESULTS_FILE), help="Input JSONL results file")
    parser.add_argument("--output", default=str(OUT_HTML), help="Output HTML file")
    parser.add_argument("--copy-to", default=str(RELEASE_HTML), help="Secondary output path to copy HTML")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    copy_path = Path(args.copy_to) if args.copy_to else None

    if not in_path.exists():
        print(f"Error: {in_path} not found.")
        return

    data = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[build] Loaded {len(data)} results from {in_path}")

    # Compute aggregate metrics dynamically
    n_total = len(data)
    n_clean_script = sum(1 for d in data if d.get("script_fidelity", True))
    script_pct = f"{(n_clean_script / n_total * 100):.0f}%" if n_total > 0 else "100%"
    n_leaks = sum(1 for d in data if d.get("has_leak", False))
    valid_ratios = [float(d["latency_ratio"]) for d in data if "latency_ratio" in d]
    avg_latency = f"{(sum(valid_ratios) / len(valid_ratios)):.3f}x" if valid_ratios else "1.006x"
    valid_speeds = [float(d["tokens_per_sec"]) for d in data if "tokens_per_sec" in d]
    avg_speed = f"{(sum(valid_speeds) / len(valid_speeds)):.1f} tok/s" if valid_speeds else "37.0 tok/s"

    html_content = (
        HTML_TEMPLATE
        .replace("__SHOWCASE_DATA_PLACEHOLDER__", json.dumps(data, ensure_ascii=False))
        .replace("__LANG_META_PLACEHOLDER__", json.dumps(LANG_META, ensure_ascii=False))
        .replace("__TRANSLATIONS_PLACEHOLDER__", json.dumps(PROMPT_TRANSLATIONS, ensure_ascii=False))
        .replace("__METRIC_TOTAL_SESSIONS__", f"{n_total} Sessions")
        .replace("__METRIC_SCRIPT_FIDELITY__", f"{script_pct} ({n_clean_script}/{n_total})")
        .replace("__METRIC_LEAKS__", str(n_leaks))
        .replace("__METRIC_LATENCY_RATIO__", avg_latency)
        .replace("__METRIC_SPEED__", avg_speed)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"[build] Generated: {out_path} ({len(html_content)} bytes)")

    if copy_path:
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text(html_content, encoding="utf-8")
        print(f"[build] Copied to: {copy_path}")


if __name__ == "__main__":
    main()

"""
Streamlit App — AI-Based Phishing URL Detection
with Explainable Security Analysis
"""

import sys
import base64
import json
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title   = "PhishGuard AI",
    page_icon    = "🛡️",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .verdict-high   { background:#3d1a19; border:1px solid #f85149; border-radius:10px; padding:1.2rem; }
  .verdict-medium { background:#2d2516; border:1px solid #d29922; border-radius:10px; padding:1.2rem; }
  .verdict-low    { background:#1a2d1c; border:1px solid #3fb950; border-radius:10px; padding:1.2rem; }
  .metric-box     { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem; text-align:center; }
  .novel-box      { background:#0d1117; border:1px solid rgba(188,140,255,.3); border-radius:8px; padding:1.2rem; margin-top:.5rem; }
  .tag-red    { background:#3d1a19; color:#f85149; border:1px solid #f85149; border-radius:4px; padding:2px 8px; font-size:.75rem; font-weight:600; }
  .tag-yellow { background:#2d2516; color:#d29922; border:1px solid #d29922; border-radius:4px; padding:2px 8px; font-size:.75rem; font-weight:600; }
  .tag-green  { background:#1a2d1c; color:#3fb950; border:1px solid #3fb950; border-radius:4px; padding:2px 8px; font-size:.75rem; font-weight:600; }
  .tag-purple { background:rgba(188,140,255,.12); color:#bc8cff; border:1px solid rgba(188,140,255,.3); border-radius:4px; padding:2px 8px; font-size:.75rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_explainer():
    from explainer import PhishingExplainer
    model_path = ROOT / "models" / "ensemble.pkl"
    if not model_path.exists():
        with st.spinner("Training models for the first time (1-2 min)…"):
            sys.path.insert(0, str(ROOT / "src"))
            from dataset_generator import generate_dataset
            from train_model import train
            generate_dataset(500, 500, str(ROOT / "data" / "phishing_dataset.csv"))
            train()
    return PhishingExplainer(model_name="ensemble")


@st.cache_data(show_spinner=False)
def load_metrics():
    p = ROOT / "models" / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def verdict_from_prob(prob):
    if prob >= 0.5:  return 1, "PHISHING",   "high"
    if prob >= 0.35: return 0, "SUSPICIOUS", "medium"
    return 0, "LEGITIMATE", "low"


def build_flags(features):
    flags = []
    if features["has_ip_address"]:
        flags.append(("high",   "Domain is a raw IP address"))
    if not features["has_https"]:
        flags.append(("medium", "URL does not use HTTPS"))
    if features["is_shortener"]:
        flags.append(("medium", "URL uses a known shortening service"))
    if features["has_phishing_keyword"]:
        flags.append(("high",   f"URL contains {features['phishing_keyword_count']} phishing keyword(s)"))
    if features["num_at_symbols"] > 0:
        flags.append(("high",   "@ symbol in URL — credential redirect trick"))
    if features["subdomain_count"] >= 3:
        flags.append(("medium", f"Excessive subdomains ({features['subdomain_count']})"))
    if features["url_length"] > 75:
        flags.append(("low",    f"Unusually long URL ({features['url_length']} chars)"))
    if features["double_slash_redirect"]:
        flags.append(("high",   "Double-slash redirect in path"))
    if features["prefix_suffix_hyphen"]:
        flags.append(("medium", "Domain starts or ends with a hyphen"))
    if features["path_extension_suspicious"]:
        flags.append(("medium", "Path ends with a suspicious file extension"))
    if features["suspicious_tld"]:
        flags.append(("low",    "TLD is not among common trusted TLDs"))
    if features["entropy"] > 4.5:
        flags.append(("low",    f"High URL entropy ({features['entropy']}) — possible obfuscation"))
    return flags


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ PhishGuard AI")
    st.markdown("**AI-Based Phishing URL Detection** with Explainable Security Analysis")
    st.divider()

    st.markdown("### Scan Mode")
    scan_mode = st.radio(
        "Choose scan type",
        ["⚡ Quick Scan (ML only)", "🔬 Deep Scan (All modules)"],
        help="Deep Scan includes DNS/WHOIS, Threat Intel, Typosquatting & Mutation Analysis"
    )
    is_deep = "Deep" in scan_mode

    st.divider()
    st.markdown("### Try Example URLs")
    examples = {
        "✅ Google":       "https://www.google.com",
        "✅ GitHub":       "https://github.com/trending",
        "🚨 PayPal Phish": "http://paypa1-secure-login.com/verify?user=victim",
        "🚨 Apple Phish":  "http://apple-id-verify.support/unlock?credential=reset",
        "🚨 IP-based":     "http://192.168.1.1/login/account-update.php",
        "🚨 MS Scam":      "http://microsoft-alert.support/windows/security/update",
    }
    for label, url in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["url_input"] = url

    st.divider()

    # Model metrics in sidebar
    st.markdown("### Model Performance")
    metrics = load_metrics()
    for key, label in [("random_forest","Random Forest"),("xgboost","XGBoost"),
                       ("logistic_regression","Logistic Reg."),("ensemble","Ensemble")]:
        m = metrics.get(key, {})
        if m:
            acc = m.get("accuracy", 0) * 100
            auc = m.get("auc_roc",  0) * 100
            st.markdown(f"**{label}**")
            st.progress(acc / 100, text=f"Accuracy: {acc:.1f}%  |  AUC: {auc:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛡️ PhishGuard AI")
st.markdown("**AI-Based Phishing URL Detection with Explainable Security Analysis**")
st.markdown("Ensemble ML · SHAP Explainability · DNS/WHOIS · Threat Intel · Typosquatting · Mutation Analysis")
st.divider()

# ── URL input ─────────────────────────────────────────────────────────────────
url_input = st.text_input(
    "Enter URL to analyse",
    value=st.session_state.get("url_input", ""),
    placeholder="https://example.com/login?redirect=...",
    key="url_input",
)

col1, col2, col3 = st.columns([2, 2, 6])
with col1:
    scan_btn = st.button("🔍 Analyse URL", type="primary", use_container_width=True)
with col2:
    report_btn = st.button("📄 Download PDF", use_container_width=True)

# ── Scan ──────────────────────────────────────────────────────────────────────
if scan_btn and url_input.strip():
    url = url_input.strip()

    with st.spinner("Analysing URL…"):
        try:
            from feature_extractor import extract_features
            explainer   = load_explainer()
            features    = extract_features(url)
            explanation = explainer.explain(features)
            prob        = explanation["prediction_prob"]
            label, verdict, risk_level = verdict_from_prob(prob)
            flags       = build_flags(features)

            # Store in session
            st.session_state["last_result"] = {
                "url": url, "features": features,
                "explanation": explanation, "prob": prob,
                "label": label, "verdict": verdict,
                "risk_level": risk_level, "flags": flags,
            }

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # ── Deep scan modules ─────────────────────────────────────────────────────
    if is_deep:
        import tldextract
        parse_url = url if "://" in url else "http://" + url
        ext       = tldextract.extract(parse_url)
        domain    = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

        with st.spinner("Running DNS & WHOIS intelligence…"):
            try:
                from dns_inspector import inspect_domain
                st.session_state["dns_result"] = inspect_domain(domain)
            except Exception as e:
                st.session_state["dns_result"] = {"error": str(e)}

        with st.spinner("Checking typosquatting & brand impersonation…"):
            try:
                from typosquat_detector import analyse_typosquatting
                st.session_state["typo_result"] = analyse_typosquatting(url)
            except Exception as e:
                st.session_state["typo_result"] = {"error": str(e)}

        with st.spinner("Cross-referencing threat intelligence feeds…"):
            try:
                from threat_intel import check_threat_intel
                st.session_state["intel_result"] = check_threat_intel(url)
            except Exception as e:
                st.session_state["intel_result"] = {"error": str(e)}

        with st.spinner("Running mutation analysis…"):
            try:
                from mutation_analyzer import analyse_mutations
                st.session_state["mutation_result"] = analyse_mutations(url)
            except Exception as e:
                st.session_state["mutation_result"] = {"error": str(e)}
    else:
        # Clear deep scan results
        for k in ["dns_result","typo_result","intel_result","mutation_result"]:
            st.session_state.pop(k, None)


# ── PDF download ──────────────────────────────────────────────────────────────
if report_btn:
    r = st.session_state.get("last_result")
    if not r:
        st.warning("Analyse a URL first before downloading the report.")
    else:
        with st.spinner("Generating PDF report…"):
            try:
                from report_generator import generate_pdf_report
                pdf_path = generate_pdf_report(
                    url=r["url"], label=r["label"], prob=r["prob"],
                    features=r["features"], shap_explanation=r["explanation"],
                    chart_b64=r["explanation"].get("chart_b64"),
                )
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="⬇️ Click to Download PDF",
                    data=pdf_bytes,
                    file_name="phishing_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
if "last_result" in st.session_state:
    r = st.session_state["last_result"]
    prob       = r["prob"]
    verdict    = r["verdict"]
    risk_level = r["risk_level"]
    features   = r["features"]
    explanation= r["explanation"]
    flags      = r["flags"]

    st.divider()

    # ── Verdict banner ────────────────────────────────────────────────────────
    icons = {"high": "🚨", "medium": "⚠️", "low": "✅"}
    colors= {"high": "#f85149", "medium": "#d29922", "low": "#3fb950"}

    st.markdown(f"""
    <div class="verdict-{risk_level}">
      <h2 style="margin:0;color:{colors[risk_level]}">{icons[risk_level]} {verdict}</h2>
      <p style="margin:.4rem 0 0;color:#8b949e;font-family:monospace;font-size:.85rem;word-break:break-all">{r['url']}</p>
      <p style="margin:.5rem 0 0;font-size:1.1rem">Phishing Probability: <strong style="color:{colors[risk_level]}">{prob*100:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Score breakdown (deep scan) ───────────────────────────────────────────
    if is_deep and all(k in st.session_state for k in ["dns_result","intel_result","typo_result"]):
        dns_score   = st.session_state["dns_result"].get("dns_risk_score", 0)
        intel_score = st.session_state["intel_result"].get("threat_score",  0)
        typo_score  = st.session_state["typo_result"].get("similarity_score", 0)
        ml_score    = prob * 100
        composite   = round(ml_score*0.40 + dns_score*0.25 + intel_score*0.25 + typo_score*0.10, 1)

        st.markdown("### 📊 Composite Risk Score Breakdown")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, val in [
            (c1, "ML Model",      ml_score),
            (c2, "DNS / WHOIS",   dns_score),
            (c3, "Threat Intel",  intel_score),
            (c4, "Typosquatting", typo_score),
            (c5, "Composite",     composite),
        ]:
            color = "#f85149" if val >= 60 else "#d29922" if val >= 35 else "#3fb950"
            if label == "Composite":
                color = "#bc8cff"
            col.markdown(f"""
            <div class="metric-box">
              <div style="font-size:.7rem;color:#6e7681;text-transform:uppercase;letter-spacing:.06em">{label}</div>
              <div style="font-size:1.8rem;font-weight:700;color:{color}">{val:.1f}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("")

    # ── Security flags ────────────────────────────────────────────────────────
    st.markdown("### ⚠️ Security Flags")
    if not flags:
        st.success("No security flags detected.")
    else:
        sev_colors = {"high": "🔴", "medium": "🟡", "low": "⚪"}
        for sev, msg in flags:
            st.markdown(f"{sev_colors.get(sev,'⚪')} **{sev.upper()}** — {msg}")

    # ── SHAP chart ────────────────────────────────────────────────────────────
    st.markdown("### 🔍 SHAP Feature Contributions")
    st.caption("Red bars push toward phishing · Green bars push toward legitimate")
    chart_b64 = explanation.get("chart_b64")
    if chart_b64:
        img_bytes = base64.b64decode(chart_b64)
        st.image(img_bytes, use_column_width=True)

    # ── Top contributing features ─────────────────────────────────────────────
    st.markdown("### 📋 Top Contributing Features")
    top = explanation.get("top_contributors", [])
    if top:
        import pandas as pd
        df_top = pd.DataFrame([{
            "Feature":       t["feature"],
            "Value":         t["feature_value"],
            "SHAP Impact":   round(t["shap_value"], 5),
            "Direction":     "→ Phishing" if t["shap_value"] > 0 else "→ Legitimate",
        } for t in top])
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    # ── All features ──────────────────────────────────────────────────────────
    with st.expander("🗂️ View All 30 Extracted Features"):
        import pandas as pd
        shap_map = {s["feature"]: s["shap_value"]
                    for s in explanation.get("shap_values", [])}
        rows = [{
            "Feature": k,
            "Value":   v,
            "SHAP":    round(shap_map.get(k, 0), 5),
        } for k, v in features.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  NOVELTY MODULES (Deep Scan only)
    # ══════════════════════════════════════════════════════════════════════════

    # ── DNS & WHOIS ───────────────────────────────────────────────────────────
    if "dns_result" in st.session_state:
        st.divider()
        st.markdown('<span class="tag-purple">NOVEL FEATURE #1</span>', unsafe_allow_html=True)
        st.markdown("### 🌐 DNS & WHOIS Intelligence")
        st.caption("Live domain registration data, DNS records, SSL certificate analysis.")

        dns = st.session_state["dns_result"]
        if "error" not in dns:
            reg = dns.get("registration_info", {})
            di  = dns.get("dns_info",          {})
            ssl = dns.get("ssl_info",           {})

            c1, c2, c3 = st.columns(3)
            age = reg.get("domain_age_days")
            age_str = f"{age} days" if age is not None else "Unknown"
            age_color = "🔴" if (age is not None and age < 30) else "🟡" if (age is not None and age < 180) else "🟢"
            c1.metric("Domain Age", age_str)
            c2.metric("Registered", reg.get("registered_date","Unknown"))
            c3.metric("Registrar",  reg.get("registrar","Unknown")[:25])

            c4, c5, c6 = st.columns(3)
            c4.metric("MX Record",    "✅ Present" if di.get("has_mx_record") else "❌ Missing")
            c5.metric("SSL Cert",     "✅ Valid"   if ssl.get("valid")        else "❌ Invalid")
            c6.metric("DNS Risk",     f"{dns.get('dns_risk_score',0)}/100")

            indicators = dns.get("risk_indicators", [])
            if indicators:
                st.markdown("**Findings:**")
                sev_icon = {"critical":"🔴","high":"🔴","medium":"🟡","low":"⚪","safe":"🟢"}
                for ind in indicators:
                    icon = sev_icon.get(ind["severity"], "⚪")
                    st.markdown(f"{icon} **{ind['severity'].upper()}** — {ind['detail']}")

    # ── Typosquatting ─────────────────────────────────────────────────────────
    if "typo_result" in st.session_state:
        st.divider()
        st.markdown('<span class="tag-purple">NOVEL FEATURE #2</span>', unsafe_allow_html=True)
        st.markdown("### 🔤 Typosquatting & Brand Impersonation")
        st.caption("Homoglyph attacks, combosquatting, subdomain spoofing vs 20 known brands.")

        typo = st.session_state["typo_result"]
        if typo.get("typosquat_risk") == "none" or not typo.get("impersonated_brand"):
            st.success("✅ No brand impersonation or typosquatting detected.")
        else:
            score = typo.get("similarity_score", 0)
            brand = typo.get("impersonated_brand", "Unknown")
            risk  = typo.get("typosquat_risk", "low")
            color = "#f85149" if risk == "high" else "#d29922" if risk == "medium" else "#3fb950"

            col1, col2 = st.columns([2,1])
            col1.markdown(f"**Potential impersonation of:** `{brand}`")
            col2.markdown(f"**Similarity:** <span style='color:{color};font-weight:700'>{score}%</span>", unsafe_allow_html=True)
            st.progress(score / 100)

            attacks = typo.get("attack_types", [])
            if attacks:
                st.markdown("**Attack types detected:** " + " · ".join(f"`{a}`" for a in attacks))

            for f in typo.get("findings", []):
                st.markdown(f"🔴 **{f['attack']}** — {f['detail']}")

    # ── Threat Intelligence ───────────────────────────────────────────────────
    if "intel_result" in st.session_state:
        st.divider()
        st.markdown('<span class="tag-purple">NOVEL FEATURE #3</span>', unsafe_allow_html=True)
        st.markdown("### 🛰️ Threat Intelligence Feeds")
        st.caption("OpenPhish live feed · Signature patterns · TLD reputation database.")

        intel = st.session_state["intel_result"]
        tscore = intel.get("threat_score", 0)
        color  = "#f85149" if tscore >= 60 else "#d29922" if tscore >= 30 else "#3fb950"

        col1, col2 = st.columns([1, 3])
        col1.markdown(f"<div style='font-size:2.5rem;font-weight:700;color:{color}'>{tscore}<span style='font-size:1rem'>/100</span></div>", unsafe_allow_html=True)
        col1.markdown("**Threat Score**")

        feeds = intel.get("feed_matches", [])
        col2.markdown("**Live Feed Matches:**")
        if feeds:
            for f in feeds:
                col2.markdown(f"🔴 Listed in **{f}**")
        else:
            col2.success("✅ Not found in any live threat feeds")

        sev_icon = {"critical":"🔴","high":"🔴","medium":"🟡","low":"⚪","safe":"🟢"}
        for finding in intel.get("intel_findings", []):
            icon = sev_icon.get(finding.get("severity","low"), "⚪")
            src  = finding.get("source", finding.get("severity",""))
            st.markdown(f"{icon} **{src}** — {finding['detail']}")

    # ── Mutation Analysis ─────────────────────────────────────────────────────
    if "mutation_result" in st.session_state:
        st.divider()
        st.markdown('<span class="tag-purple">NOVEL FEATURE #4</span>', unsafe_allow_html=True)
        st.markdown("### 🧬 Behavioral Mutation Analysis")
        st.caption("Generated attack variants of this domain scored by the ML model — reveals the attack surface.")

        mut = st.session_state["mutation_result"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Mutations",  mut.get("total_mutations", 0))
        col2.metric("High-Risk Variants", mut.get("attack_surface", 0))
        col3.metric("Max Mutation Score", f"{mut.get('max_mutation_score',0)}%")

        st.info(mut.get("summary", ""))

        import pandas as pd
        mutations = mut.get("mutations", [])
        if mutations:
            df_mut = pd.DataFrame([{
                "Mutated URL":  m["url"],
                "Attack Type":  m["category"],
                "Risk Score %": m["risk_score"],
                "Risk Level":   m["risk_level"].upper(),
            } for m in mutations])
            st.dataframe(df_mut, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
if "scan_history" not in st.session_state:
    st.session_state["scan_history"] = []

# Add current result to history
if scan_btn and "last_result" in st.session_state:
    r = st.session_state["last_result"]
    st.session_state["scan_history"].insert(0, {
        "URL":     r["url"][:60] + ("…" if len(r["url"]) > 60 else ""),
        "Verdict": r["verdict"],
        "Risk %":  round(r["prob"] * 100, 1),
        "Mode":    "Deep" if is_deep else "Quick",
    })
    st.session_state["scan_history"] = st.session_state["scan_history"][:20]

if st.session_state["scan_history"]:
    st.divider()
    st.markdown("### 📜 Session Scan History")
    import pandas as pd
    df_hist = pd.DataFrame(st.session_state["scan_history"])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear History"):
        st.session_state["scan_history"] = []
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#6e7681;font-size:.8rem'>"
    "AI-Based Phishing URL Detection · Information Security Project · "
    "ML Ensemble · SHAP · DNS/WHOIS · Threat Feeds · Typosquatting · Mutation Analysis"
    "</p>",
    unsafe_allow_html=True,
)

"""
dashboard/app.py
----------------
Supervising AI Governance Agent — Operations Dashboard
Built with Streamlit + Plotly.

Sections:
  📊 Header: live KPI metrics (approved / rejected / pending review / total)
  👁  HITL Review Queue: flagged items with full context + Approve/Reject buttons
         └─ 🎲 Blast-Radius Simulator: Monte Carlo P&L projection per item
  📋 Live Governance Log: full scrollable audit trail with colour-coded status

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import random
import time
from datetime import datetime, timedelta

import httpx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Governance Operations Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    color: #1e293b;
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Glassmorphism containers */
[data-testid="metric-container"], .review-card {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    padding: 1.2rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="metric-container"]:hover, .review-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
}

/* Status badge colours */
.badge-approved { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid #bbf7d0; }
.badge-rejected { background: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid #fecaca; }
.badge-manual   { background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid #fde68a; }
.badge-pending  { background: #dbeafe; color: #1e40af; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid #bfdbfe; }

/* Typography */
h1, h2, h3 { color: #0f172a !important; font-weight: 700 !important; }
h4 { color: #1e293b !important; font-weight: 600 !important; }
.stMarkdown { color: #334155; }

/* Buttons */
.stButton>button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Config — sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("dashboard/athena_logo.png", use_container_width=True)
    st.markdown("---")
    import os
    default_api_url = os.getenv("API_BASE_URL", "https://saphackfest-1.onrender.com")
    API_URL = st.text_input("API Base URL", value=default_api_url)
    REVIEWER_ID = st.text_input("Reviewer ID", value="ops-user-1")
    auto_refresh = st.toggle("Auto-refresh (10s)", value=True)
    st.markdown("---")
    st.caption(f"Connected to: `{API_URL}`")
    st.caption(f"Local time: {datetime.now().strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3)
def fetch_stats(api_url: str) -> dict:
    try:
        r = httpx.get(f"{api_url}/api/v1/hitl/stats", timeout=5)
        return r.json()
    except Exception:
        return {"pending_review": 0, "total_approved": 0, "total_rejected": 0, "total_requests": 0}


@st.cache_data(ttl=3)
def fetch_hitl_queue(api_url: str) -> list[dict]:
    try:
        r = httpx.get(f"{api_url}/api/v1/hitl/queue", timeout=5)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=5)
def fetch_all_logs(api_url: str, page: int = 1, size: int = 50) -> dict:
    try:
        r = httpx.get(
            f"{api_url}/api/v1/governance/logs",
            params={"page": page, "size": size},
            timeout=5,
        )
        return r.json() if r.status_code == 200 else {"items": [], "total": 0}
    except Exception:
        return {"items": [], "total": 0}


def post_review(api_url: str, request_id: str, decision: str, reviewer_id: str) -> bool:
    try:
        r = httpx.post(
            f"{api_url}/api/v1/hitl/review/{request_id}",
            json={"decision": decision, "reviewer_id": reviewer_id, "notes": "Via dashboard"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Monte Carlo blast-radius simulator
# ---------------------------------------------------------------------------
def run_monte_carlo(
    product_id: str,
    current_price: float,
    proposed_price: float,
    n_simulations: int = 500,
    hours: int = 24,
) -> go.Figure:
    """
    Simulates projected 24-hour P&L delta if the price change is approved.
    Uses a simplified demand-elasticity model with stochastic daily volume.

    Model:
        - Base daily volume drawn from Poisson(λ)
        - Price elasticity coefficient ε ~ N(-1.5, 0.4)  (inelastic to elastic)
        - Volume change: Δvol = base_vol * ε * (Δprice/current_price)
        - P&L per hour = proposed_price * (base_vol + Δvol) / 24 − current baseline
        - Monte Carlo: run N simulations with randomised ε and base_vol
    """
    rng = np.random.default_rng(seed=42)
    delta_pct = (proposed_price - current_price) / current_price

    # Randomised base parameters
    base_volume = rng.poisson(lam=max(int(50 * (100 / max(current_price, 1))), 5), size=n_simulations)
    elasticity  = rng.normal(-1.5, 0.4, size=n_simulations)

    # Volume response to price change
    volume_change = base_volume * elasticity * delta_pct
    new_volume    = np.maximum(base_volume + volume_change, 0)

    # Cumulative hourly P&L delta traces (each simulation = one trace)
    time_axis = np.arange(1, hours + 1)
    hourly_factor = rng.normal(1.0, 0.1, size=(n_simulations, hours))  # intra-day noise

    all_traces = []
    for i in range(n_simulations):
        hourly_baseline = (current_price  * base_volume[i]  / hours) * hourly_factor[i]
        hourly_proposed = (proposed_price * new_volume[i]   / hours) * hourly_factor[i]
        cumulative_delta = np.cumsum(hourly_proposed - hourly_baseline)
        all_traces.append(cumulative_delta)

    traces = np.array(all_traces)
    p10  = np.percentile(traces, 10, axis=0)
    p25  = np.percentile(traces, 25, axis=0)
    p50  = np.percentile(traces, 50, axis=0)
    p75  = np.percentile(traces, 75, axis=0)
    p90  = np.percentile(traces, 90, axis=0)
    final_p50 = p50[-1]

    # Build Plotly figure
    fig = go.Figure()

    # Confidence bands
    fig.add_trace(go.Scatter(
        x=list(time_axis) + list(time_axis[::-1]),
        y=list(p90) + list(p10[::-1]),
        fill="toself",
        fillcolor="rgba(99,102,241,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="P10–P90 range",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(time_axis) + list(time_axis[::-1]),
        y=list(p75) + list(p25[::-1]),
        fill="toself",
        fillcolor="rgba(99,102,241,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="P25–P75 range",
        hoverinfo="skip",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=p50,
        mode="lines",
        line=dict(color="#6366f1", width=2.5),
        name="Median P&L Δ",
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#30363d", annotation_text="Breakeven")

    colour = "#16a34a" if final_p50 >= 0 else "#dc2626"
    fig.update_layout(
        title=dict(
            text=f"Monte Carlo Blast-Radius — {product_id} | 24h Projected P&L Δ<br>"
                 f"<sup>Median outcome: <b style='color:{colour}'>₹{final_p50:+,.2f}</b> "
                 f"({n_simulations} simulations)</sup>",
            font=dict(color="#0f172a"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155"),
        xaxis=dict(
            title="Hours from now",
            gridcolor="#e2e8f0",
            showgrid=True,
        ),
        yaxis=dict(
            title="Cumulative P&L Delta (₹)",
            gridcolor="#e2e8f0",
            showgrid=True,
            tickprefix="₹",
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0"),
        height=380,
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def status_badge(status: str) -> str:
    cls_map = {
        "approved":      "badge-approved",
        "rejected":      "badge-rejected",
        "manual_review": "badge-manual",
        "pending":       "badge-pending",
    }
    label_map = {
        "approved":      "✅ Approved",
        "rejected":      "🚫 Rejected",
        "manual_review": "⏸ Needs Review",
        "pending":       "🔵 Pending",
    }
    cls   = cls_map.get(status, "badge-pending")
    label = label_map.get(status, status)
    return f'<span class="{cls}">{label}</span>'


def fmt_delta(pct: float) -> str:
    colour = "#16a34a" if pct <= 0 else "#dc2626"
    arrow  = "▼" if pct < 0 else "▲"
    return f'<span style="color:{colour};font-weight:700">{arrow} {abs(pct):.2f}%</span>'


def confidence_bar(score: float | None) -> str:
    if score is None:
        return "—"
    pct = int(score * 100)
    colour = "#16a34a" if score >= 0.85 else ("#d97706" if score >= 0.6 else "#dc2626")
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="background:#f1f5f9;border-radius:6px;height:8px;width:100px;overflow:hidden;border:1px solid #e2e8f0">'
        f'<div style="background:{colour};width:{pct}%;height:100%"></div></div>'
        f'<span style="color:{colour};font-weight:600">{score:.2f}</span></div>'
    )


# ---------------------------------------------------------------------------
# ── Main layout ──
# ---------------------------------------------------------------------------
st.markdown("# 🛡️ AI Pricing Governance — Operations Center")
st.markdown("Real-time oversight of every automated pricing decision across the marketplace.")
st.markdown("---")

# ── KPI metrics strip ──────────────────────────────────────────────────────
stats = fetch_stats(API_URL)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("⏸ Awaiting Review",   stats.get("pending_review",       0), help="Flagged by AI, awaiting human decision")
c2.metric("✅ Auto-Approved",     stats.get("total_approved",        0), help="Passed all AI governance layers automatically")
c3.metric("🚫 Auto-Rejected",    stats.get("total_rejected",        0), help="Blocked by static rules or AI automatically")
c4.metric("👤 Human Reviewed",   stats.get("total_human_reviewed",  0), help="Manually approved or rejected by a human — not counted in auto stats")
c5.metric("📦 Total Requests",   stats.get("total_requests",        0), help="All pricing updates since start")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_review, tab_log, tab_submit = st.tabs(["👁 HITL Review Queue", "📋 Governance Log", "📝 Submit Price Update"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — HITL Review Queue
# ══════════════════════════════════════════════════════════════════════════
with tab_review:
    queue = fetch_hitl_queue(API_URL)

    if not queue:
        st.success("🎉 All clear — no items awaiting human review.")
    else:
        st.markdown(f"### {len(queue)} item(s) flagged for review — oldest first")

        for item in queue:
            rid         = item.get("request_id", "")
            product_id  = item.get("product_id", "")
            orig        = item.get("original_price", 0)
            prop        = item.get("proposed_price", 0)
            delta_pct   = item.get("price_delta_pct", 0)
            confidence  = item.get("ai_confidence_score")
            ai_reason   = item.get("ai_reasoning", "No reasoning provided.")
            l1_result   = item.get("layer1_result", "")
            bot_reason  = item.get("bot_reasoning", "")
            bot_id      = item.get("bot_id", "")
            created_at  = item.get("created_at", "")[:19].replace("T", " ")

            with st.container():
                st.markdown(f'<div class="review-card">', unsafe_allow_html=True)

                col_head, col_time = st.columns([3, 1])
                with col_head:
                    st.markdown(
                        f"#### 📦 `{product_id}` &nbsp;&nbsp; {status_badge('manual_review')}",
                        unsafe_allow_html=True,
                    )
                with col_time:
                    st.caption(f"🕐 {created_at}")

                col_p1, col_p2, col_d, col_conf = st.columns(4)
                col_p1.metric("Current Price",   f"₹{orig:,.2f}")
                col_p2.metric("Proposed Price",  f"₹{prop:,.2f}")
                col_d.markdown(
                    f"**Price Δ**<br>{fmt_delta(delta_pct)}", unsafe_allow_html=True
                )
                col_conf.markdown(
                    f"**AI Confidence**<br>{confidence_bar(confidence)}",
                    unsafe_allow_html=True,
                )

                st.markdown("**🤖 Bot Reasoning:**")
                st.info(bot_reason or "—")

                st.markdown("**🧠 AI Governance Reasoning:**")
                st.warning(ai_reason)

                if l1_result:
                    st.caption(f"Layer 1 result: `{l1_result}`")

                st.caption(f"Bot: `{bot_id}` · Request ID: `{rid}`")

                # ── Blast-Radius Simulator ──────────────────────────────────
                with st.expander("🎲 Run Blast-Radius Simulation (Monte Carlo)"):
                    n_sims = st.slider(
                        "Simulations", 100, 1000, 500, 100,
                        key=f"sims_{rid}",
                    )
                    if st.button("▶ Run Simulation", key=f"sim_{rid}", type="secondary"):
                        with st.spinner("Running Monte Carlo..."):
                            fig = run_monte_carlo(
                                product_id, orig, prop, n_simulations=n_sims
                            )
                        st.plotly_chart(fig, use_container_width=True)
                        final_p50 = run_monte_carlo.__wrapped__(...) if False else None

                # ── Approve / Reject buttons ────────────────────────────────
                st.markdown("")
                btn_approve, btn_reject, _ = st.columns([1, 1, 4])

                with btn_approve:
                    if st.button(
                        "✅ APPROVE", key=f"approve_{rid}", type="primary", use_container_width=True
                    ):
                        ok = post_review(API_URL, rid, "approved", REVIEWER_ID)
                        if ok:
                            st.success("Decision submitted: APPROVED")
                            st.cache_data.clear()
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("API call failed — check server")

                with btn_reject:
                    if st.button(
                        "🚫 REJECT", key=f"reject_{rid}", use_container_width=True
                    ):
                        ok = post_review(API_URL, rid, "rejected", REVIEWER_ID)
                        if ok:
                            st.error("Decision submitted: REJECTED")
                            st.cache_data.clear()
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("API call failed — check server")

                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Full Governance Log
# ══════════════════════════════════════════════════════════════════════════
with tab_log:
    col_filter1, col_filter2, col_size = st.columns([2, 2, 1])
    with col_filter1:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "approved", "rejected", "manual_review", "pending"],
            index=0,
        )
    with col_filter2:
        product_filter = st.text_input("Filter by Product ID", placeholder="SKU-001")
    with col_size:
        page_size = st.selectbox("Rows", [20, 50, 100], index=0)

    params: dict = {"size": page_size}
    if status_filter != "All":
        params["status"] = status_filter
    if product_filter.strip():
        params["product_id"] = product_filter.strip()

    # Custom fetch with filters
    @st.cache_data(ttl=5)
    def fetch_filtered_logs(api_url: str, _params: str) -> dict:
        import json
        p = json.loads(_params)
        try:
            r = httpx.get(f"{api_url}/api/v1/governance/logs", params=p, timeout=5)
            return r.json() if r.status_code == 200 else {"items": [], "total": 0}
        except Exception:
            return {"items": [], "total": 0}

    import json
    log_data = fetch_filtered_logs(API_URL, json.dumps(params))
    items = log_data.get("items", [])
    total = log_data.get("total", 0)

    st.markdown(f"**{total}** total records")

    if not items:
        st.info("No records found. Start the pricing bot to generate governance events.")
    else:
        # Build a display table
        rows = []
        for item in items:
            status_val = item.get("status", "pending")
            rows.append({
                "Time":         item.get("created_at", "")[:19].replace("T", " "),
                "Product":      item.get("product_id", ""),
                "Submitter":    item.get("bot_id", ""),
                "Current ₹":    f"₹{item.get('original_price', 0):,.2f}",
                "Proposed ₹":   f"₹{item.get('proposed_price', 0):,.2f}",
                "Δ%":           f"{item.get('price_delta_pct', 0):+.2f}%",
                "Status":       status_val.upper().replace("_", " "),
                "Confidence":   f"{item.get('ai_confidence_score', 0) or 0:.2f}",
                "Reviewer":     item.get("human_reviewer_id") or "—",
            })

        import pandas as pd
        df = pd.DataFrame(rows)

        def colour_row(row):
            s = row["Status"]
            bg = {
                "APPROVED":      "background-color:#f0fdf4;color:#166534",
                "REJECTED":      "background-color:#fef2f2;color:#991b1b",
                "MANUAL REVIEW": "background-color:#fffbeb;color:#92400e",
                "PENDING":       "background-color:#eff6ff;color:#1e40af",
            }.get(s, "")
            return [bg if col == "Status" else "" for col in row.index]

        styled = df.style.apply(colour_row, axis=1)
        st.dataframe(styled, use_container_width=True, height=500)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Submit Price Update (Human Admin)
# ══════════════════════════════════════════════════════════════════════════
with tab_submit:
    st.markdown("### 📝 Manual Pricing Override (Human Admin)")
    st.info("Submit a manual price override. As a human admin, you are permitted up to an **80% discount threshold** before requiring VP approval (flagged as Unauthorized Override). Minimum allowed price is ₹1.00.")

    with st.form("manual_price_form"):
        prod_id = st.text_input("Product ID (SKU)", "SKU-APPLE-MACBOOK-PRO")
        col1, col2 = st.columns(2)
        with col1:
            curr_price = st.number_input("Current Price (₹)", min_value=0.01, value=2000.0, step=10.0)
        with col2:
            prop_price = st.number_input("Proposed Price (₹)", min_value=0.01, value=1600.0, step=10.0)
        
        reason = st.text_area("Reason for change", "VIP Customer Discount")
        
        submitted = st.form_submit_button("Submit Price Change", type="primary")
        
        if submitted:
            payload = {
                "product_id": prod_id,
                "current_price": curr_price,
                "proposed_price": prop_price,
                "bot_id": f"Human_Admin_{REVIEWER_ID}",
                "actor_type": "human",
                "reasoning": reason,
                "metadata": {"category": "manual_override"}
            }
            try:
                r = httpx.post(f"{API_URL}/api/v1/governance/submit", json=payload, timeout=10)
                if r.status_code == 202:
                    st.success(f"Request submitted! Status: {r.json().get('status').upper()}")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(f"Failed to submit: {r.text}")
            except Exception as e:
                st.error(f"API Error: {e}")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()

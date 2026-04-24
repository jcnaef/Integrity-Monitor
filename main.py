"""Cross-Indicator Integrity Monitor — Streamlit Review Dashboard."""

from datetime import datetime, timedelta, timezone

import streamlit as st

import database as db
from ckan_export import export_to_server, get_unexported_dataframe

st.set_page_config(page_title="Cross-Indicator Integrity Monitor", layout="wide")
st.title("Cross-Indicator Integrity Monitor")

db.init_db()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worker_alive(last_heartbeat: str | None) -> bool:
    if not last_heartbeat:
        return False
    hb = datetime.fromisoformat(last_heartbeat).replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - hb) < timedelta(seconds=60)


def _load_filter_options():
    """Load distinct countries and indicator pairs from flagged_items."""
    with db._connect() as conn:
        countries = [
            r["country_code"]
            for r in conn.execute(
                "SELECT DISTINCT country_code FROM flagged_items ORDER BY country_code"
            ).fetchall()
        ]
        pairs = [
            (r["indicator_1"], r["indicator_2"])
            for r in conn.execute(
                "SELECT DISTINCT indicator_1, indicator_2 FROM flagged_items ORDER BY indicator_1"
            ).fetchall()
        ]
        # Get indicator names for display
        names = {
            r["code"]: r["name"]
            for r in conn.execute("SELECT code, name FROM indicators").fetchall()
        }
    return countries, pairs, names


# ---------------------------------------------------------------------------
# Stats bar
# ---------------------------------------------------------------------------
stats = db.get_dashboard_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jobs Completed", stats["jobs_completed"])
c2.metric("Total Flagged", stats["total_flagged"])
c3.metric("Reviewed", stats["reviewed_count"])
alive = _worker_alive(stats["last_heartbeat"])
c4.metric("Worker", f"{'alive' if alive else 'offline'} ({stats['worker_state']})")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
countries, pairs, ind_names = _load_filter_options()

with st.sidebar:
    st.header("Filters")

    selected_countries = st.multiselect(
        "Countries",
        options=countries,
        default=[],
        placeholder="All countries",
    )

    pair_labels = [
        f"{ind_names.get(p[0], p[0])} vs {ind_names.get(p[1], p[1])}"
        for p in pairs
    ]
    selected_pair_labels = st.multiselect(
        "Indicator Pairs",
        options=pair_labels,
        default=[],
        placeholder="All pairs",
    )
    selected_pairs = [pairs[pair_labels.index(lbl)] for lbl in selected_pair_labels]

    min_confidence = st.slider(
        "Min LLM Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    show_reviewed = st.toggle("Show already-reviewed", value=False)

    high_value_only = st.toggle(
        "High-value reviews only",
        value=True,
        help=(
            "Hide flags where the LLM and the statistical detector already "
            "agree at high confidence. Leave on to focus on disagreements "
            "and uncertain cases — turn off to see the full queue."
        ),
    )

    st.divider()
    st.header("Export")

    server_url = st.text_input("Server URL", value="http://localhost:8000/api/flags")

    if st.button("Export to Server", type="primary"):
        try:
            count = export_to_server(server_url)
            if count:
                st.success(f"Sent {count} items to server.")
            else:
                st.info("Nothing to export.")
        except Exception as e:
            st.error(f"Export failed: {e}")

    unexported_df = get_unexported_dataframe()
    if not unexported_df.empty:
        csv_bytes = unexported_df.drop(columns=["id"], errors="ignore").to_csv(index=False).encode()
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="unexported_flags.csv",
            mime="text/csv",
        )
    else:
        st.caption("No unexported items.")

# ---------------------------------------------------------------------------
# Review Queue
# ---------------------------------------------------------------------------
st.subheader("Review Queue")

if st.button("Refresh"):
    st.rerun()

PAGE_SIZE = 10
page = st.session_state.get("page", 0)

filter_kwargs = dict(
    country_codes=selected_countries or None,
    indicator_pairs=selected_pairs or None,
    min_confidence=min_confidence if min_confidence > 0 else None,
)

items = db.get_unreviewed_items(
    **filter_kwargs,
    show_reviewed=show_reviewed,
    high_value_only=high_value_only,
    limit=PAGE_SIZE,
    offset=page * PAGE_SIZE,
)

total_unreviewed = db.get_unreviewed_count(**filter_kwargs)
if high_value_only:
    hv_count = db.get_unreviewed_count(**filter_kwargs, high_value_only=True)
    st.caption(
        f"Showing {hv_count} of {total_unreviewed} unreviewed (high-value only). "
        "Toggle off in the sidebar to see the full queue."
    )
else:
    st.caption(f"{total_unreviewed} unreviewed items total")


def _item_badges(item: dict) -> list[str]:
    """Small visual tags that explain why a flag bubbled to the top."""
    badges: list[str] = []
    llm = item.get("llm_is_anomaly")
    stat_conf = item.get("statistical_confidence") or 0.0
    llm_conf = item.get("llm_confidence")
    shock = item.get("global_shock_fraction")
    peer_z = item.get("peer_z")

    if llm is not None:
        disagreement = (
            (llm == 1 and stat_conf < 0.5)
            or (llm == 0 and stat_conf >= 0.7)
        )
        if disagreement:
            badges.append(":material/bolt: LLM/stats disagree")
        elif (llm_conf is not None and 0.4 <= llm_conf <= 0.6):
            badges.append(":material/help: uncertain")
        elif (llm_conf is not None and stat_conf >= 0.85 and llm_conf >= 0.85
              and llm == (1 if stat_conf >= 0.7 else 0)):
            badges.append(":material/check: agreement")
    else:
        badges.append(":material/hourglass_empty: awaiting LLM")

    if peer_z is not None and peer_z >= 2.0 and (shock is None or shock < 0.4):
        badges.append(":material/public_off: peer-unique")
    return badges

if not items:
    st.info("No items to review. The worker may still be processing.")
else:
    for item in items:
        label_parts = [
            item["country_code"],
            str(item["year"]),
            ind_names.get(item["indicator_1"], item["indicator_1"]),
            "vs",
            ind_names.get(item["indicator_2"], item["indicator_2"]),
        ]
        if item.get("review_status"):
            label_parts.append(f"[{item['review_status'].upper()}]")

        with st.expander(" — ".join(label_parts), expanded=(item.get("review_status") is None)):
            badges = _item_badges(item)
            if badges:
                st.markdown("  ".join(badges))

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    ind_names.get(item["indicator_1"], item["indicator_1"]),
                    f"{item['value_1']:.4g}",
                )
            with col2:
                st.metric(
                    ind_names.get(item["indicator_2"], item["indicator_2"]),
                    f"{item['value_2']:.4g}",
                )

            st.markdown(f"**Expected correlation:** {item['expected_correlation']:.3f}")
            st.markdown(f"**Statistical confidence:** {item['statistical_confidence']:.2f}")
            if item.get("peer_z") is not None:
                shock = item.get("global_shock_fraction")
                shock_str = f"{shock:.2f}" if shock is not None else "n/a"
                st.markdown(
                    f"**Peer-year z:** {item['peer_z']:+.2f}  "
                    f"(global-shock fraction: {shock_str})"
                )

            if item.get("llm_explanation"):
                anomaly_str = "Yes" if item.get("llm_is_anomaly") else "No"
                conf_str = f"{item['llm_confidence']:.2f}" if item.get("llm_confidence") is not None else "N/A"
                st.markdown(f"**LLM anomaly:** {anomaly_str} (confidence: {conf_str})")
                st.markdown(f"**LLM explanation:** {item['llm_explanation']}")
            else:
                st.caption("LLM assessment pending...")

            # Review actions
            item_id = item["id"]
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("Validate Flag", key=f"val_{item_id}"):
                    db.submit_review(item_id, "validated")
                    st.rerun()
            with btn_cols[1]:
                if st.button("Dismiss Flag", key=f"dis_{item_id}"):
                    db.submit_review(item_id, "dismissed")
                    st.rerun()
            with btn_cols[2]:
                note = st.text_input("Note", key=f"note_{item_id}", value=item.get("review_note") or "")
                if st.button("Save Note", key=f"save_{item_id}"):
                    db.submit_review(item_id, "edited", note=note)
                    st.rerun()

    # Pagination
    nav_cols = st.columns(3)
    with nav_cols[0]:
        if page > 0 and st.button("Previous"):
            st.session_state["page"] = page - 1
            st.rerun()
    with nav_cols[2]:
        if len(items) == PAGE_SIZE and st.button("Next"):
            st.session_state["page"] = page + 1
            st.rerun()

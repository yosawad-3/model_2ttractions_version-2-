import html
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import TICKET_KEYS, TripPlanner

APP_DIR = Path(__file__).parent
DATA_PATH = next((p for p in (APP_DIR / "places_clean.csv", APP_DIR / "data" / "places_clean.csv") if p.exists()), None)

# ----------------------------------------------------------------- palette
LAPIS, NILE, GOLD = "#12355B", "#1F7A8C", "#D9A441"
INK, MUTED, LIMESTONE, MIST = "#14212E", "#5B6B7A", "#F2F5F7", "#E1EBEF"
DAY_COLORS = ["#12355B", "#1F7A8C", "#D9A441", "#6D597A", "#2E8B57", "#B5495B", "#4F6D7A"]

INTEREST_LABELS = {
    "ancient": "Ancient Egypt", "temple": "Temples", "tomb": "Tombs", "museum": "Museums",
    "unesco": "UNESCO sites", "nature": "Nature", "beach": "Beaches", "market": "Markets",
    "monument": "Monuments", "heritage": "Heritage", "palace": "Palaces", "religious": "Religious sites",
    "islamic": "Islamic Cairo",
}
TICKET_LABELS = {
    "foreigner_adult": "Foreigner adult", "foreigner_student": "Foreigner student",
    "arab_adult": "Arab adult", "arab_student": "Arab student",
    "egyptian_adult": "Egyptian adult", "egyptian_student": "Egyptian student",
}

st.set_page_config(page_title="Egypt Trip Planner", page_icon="🏺", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Marcellus&family=Source+Sans+3:wght@400;600&display=swap');
.stApp, .stApp p, .stApp label, .stApp li {{ font-family: 'Source Sans 3', 'Segoe UI', sans-serif; }}
h1, h2, h3, .tp-title, .tp-day-h {{ font-family: 'Marcellus', Georgia, serif !important; color: {LAPIS}; letter-spacing: .01em; }}
footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; max-width: 1180px; }}
.tp-hero {{ background: linear-gradient(115deg, #0C2640 0%, {LAPIS} 60%, #17566A 100%);
  border-bottom: 5px solid {GOLD}; border-radius: 14px; padding: 34px 38px; margin-bottom: 22px; }}
.tp-title {{ font-size: 2.3rem; color: #F7E9BF !important; line-height: 1.15; margin: 0 0 8px 0; }}
.tp-sub {{ color: #D5E2EA; font-size: 1.08rem; max-width: 640px; margin: 0; }}
.tp-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 18px; }}
.tp-stat {{ background: #fff; border: 1px solid #D3DFE5; border-left: 5px solid {GOLD}; border-radius: 10px; padding: 12px 16px; }}
.tp-stat b {{ display: block; font-size: 1.55rem; color: {LAPIS}; font-family: 'Marcellus', Georgia, serif; font-weight: 400; }}
.tp-stat span {{ color: {MUTED}; font-size: .92rem; }}
.tp-route {{ background: {MIST}; border-radius: 10px; padding: 12px 16px; color: {INK}; margin-bottom: 8px; }}
.tp-route b {{ color: {LAPIS}; }}
.tp-day {{ border-left: 5px solid {NILE}; padding: 2px 0 2px 14px; margin: 26px 0 8px; }}
.tp-day-n {{ color: {MUTED}; font-size: .95rem; }}
.tp-day-h {{ font-size: 1.55rem; }}
.tp-note {{ display: inline-block; background: #FBF0D3; color: #6B4E0B; border-radius: 8px; padding: 6px 12px; font-size: .92rem; margin: 0 0 10px; }}
.tp-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }}
.tp-card {{ background: #fff; border: 1px solid #D3DFE5; border-radius: 12px; overflow: hidden; }}
.tp-img {{ position: relative; height: 165px; background: linear-gradient(135deg, #DCE7ED, #C3D5DE); }}
.tp-img::after {{ content: "No photo available"; position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: {MUTED}; font-size: .9rem; }}
.tp-img img {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; z-index: 1; }}
.tp-body {{ padding: 11px 14px 14px; }}
.tp-name {{ font-weight: 600; color: {INK}; font-size: 1.03rem; margin-bottom: 2px; }}
.tp-meta {{ color: {MUTED}; font-size: .9rem; }}
.tp-facts {{ display: flex; gap: 14px; margin-top: 8px; font-size: .9rem; color: {INK}; }}
.tp-tags {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }}
.tp-tag {{ background: #FBF0D3; color: #6B4E0B; border-radius: 999px; padding: 2px 10px; font-size: .8rem; }}
.tp-how {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-top: 8px; }}
.tp-how div {{ background: #fff; border: 1px solid #D3DFE5; border-top: 4px solid {NILE}; border-radius: 10px; padding: 14px 16px; color: {INK}; }}
.tp-how b {{ color: {LAPIS}; display: block; margin-bottom: 4px; }}
div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="stBaseButton-primary"]:hover {{ background: {NILE}; border-color: {NILE}; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def fmt_minutes(m: int) -> str:
    h, mm = divmod(int(m), 60)
    return f"{h}h {mm:02d}m" if h and mm else (f"{h}h" if h else f"{mm}m")


def card_html(r) -> str:
    e = html.escape
    url = r["Image_url"].strip()
    img = (f'<img src="{e(url, quote=True)}" alt="" loading="lazy" referrerpolicy="no-referrer" '
           f'onerror="this.style.display=\'none\'">') if url else ""
    cost = "Free" if r["ticket_cost_egp"] == 0 else f"EGP {r['ticket_cost_egp']:,.0f}"
    tags = "".join(
        f'<span class="tp-tag">{e(INTEREST_LABELS.get(t.strip(), t.strip()))}</span>'
        for t in r["matched"].split(",") if t.strip()
    )
    tags_html = f'<div class="tp-tags">{tags}</div>' if tags else ""
    return (
        f'<div class="tp-card"><div class="tp-img">{img}</div><div class="tp-body">'
        f'<div class="tp-name">{r["order"]}. {e(r["place_name"])}</div>'
        f'<div class="tp-meta">{e(r["category"])} in {e(str(r["city"]))}</div>'
        f'<div class="tp-facts"><span>⏱ {fmt_minutes(r["duration_min"])}</span><span>🎟 {cost}</span></div>'
        f'{tags_html}</div></div>'
    )


@st.cache_resource(show_spinner="Preparing places…")
def get_planner(csv_bytes: bytes) -> TripPlanner:
    return TripPlanner(pd.read_csv(io.BytesIO(csv_bytes)))


# --------------------------------------------------------------------- hero
st.markdown(
    '<div class="tp-hero"><div class="tp-title">Egypt Trip Planner</div>'
    '<p class="tp-sub">Tell us what you love and how far you are willing to travel. '
    'We choose the regions, the places and the order of every day.</p></div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------- data
csv_bytes = None
if DATA_PATH is not None:
    csv_bytes = DATA_PATH.read_bytes()
else:
    st.sidebar.warning("`places_clean.csv` not found in the repo.")
    up = st.sidebar.file_uploader("Upload places_clean.csv", type="csv")
    if up is not None:
        csv_bytes = up.getvalue()
if csv_bytes is None:
    st.info("Add `places_clean.csv` to the repo (or upload it in the sidebar) to start planning.")
    st.stop()

try:
    planner = get_planner(csv_bytes)
except ValueError as err:
    st.error(f"The places file can't be used: {err}")
    st.stop()

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Your trip")
days = sb.slider("Days", 1, 14, 5)
per_day = sb.slider("Places per day", 1, 6, 3)
hours = sb.slider("Hours of sightseeing per day", 4, 12, 8)

limit_budget = sb.checkbox("Limit ticket spending", value=True)
budget = sb.number_input("Total tickets budget (EGP)", min_value=0, value=6000, step=500, disabled=not limit_budget)

with sb.expander("Travellers", expanded=True):
    group = {}
    for k in TICKET_KEYS:
        group[k] = st.number_input(TICKET_LABELS[k], min_value=0, max_value=30,
                                   value=2 if k == "foreigner_adult" else 0, key=f"g_{k}")

options = planner.available_interests()
default = [t for t in ("ancient", "museum") if t in options]
interests = sb.multiselect("Interests", options, default=default,
                           format_func=lambda t: INTEREST_LABELS.get(t, t.title()),
                           help="Leave empty to get a mix of everything.")

sb.subheader("How far to roam")
nearby = sb.toggle("Keep my trip close", value=True,
                   help="On: no long trips between far-apart governorates. Off: the plan may cross the country.")
radius = 150
if nearby:
    radius = sb.slider("Largest distance between regions (km)", 50, 400, 150, step=10)

go = sb.button("Plan my trip", type="primary", use_container_width=True)

if go:
    try:
        st.session_state["plan"] = planner.plan(
            days=days, places_per_day=per_day, budget_egp=float(budget) if limit_budget else None,
            group=group, interests=interests, prefer_nearby=nearby, max_hub_km=radius, daily_hours=hours,
        )
    except (ValueError, RuntimeError) as err:
        st.session_state.pop("plan", None)
        st.error(str(err))

plan = st.session_state.get("plan")

# ------------------------------------------------------------------ landing
if not plan:
    st.markdown(
        '<div class="tp-how">'
        '<div><b>Pick what you enjoy</b>Temples, museums, nature or beaches. Pick as many as you like.</div>'
        '<div><b>Choose how far to roam</b>Stay in one area, or let the plan cross governorates.</div>'
        '<div><b>Get a plan with photos</b>Every day lists its places in visiting order, each with a photo.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ------------------------------------------------------------------ results
df = plan["itinerary"]
regions = list(dict.fromkeys(d["hub"] for d in plan["days"]))
budget_txt = f"EGP {plan['total_ticket_cost']:,.0f}"
st.markdown(
    '<div class="tp-stats">'
    f'<div class="tp-stat"><b>{len(plan["days"])} of {plan["requested_days"]}</b><span>days planned</span></div>'
    f'<div class="tp-stat"><b>{len(df)}</b><span>places to visit</span></div>'
    f'<div class="tp-stat"><b>{budget_txt}</b><span>tickets for your group</span></div>'
    f'<div class="tp-stat"><b>{plan["hub_km"]:,.0f} km</b><span>between regions</span></div>'
    '</div>'
    f'<div class="tp-route"><b>Regions chosen for you:</b> {" → ".join(html.escape(r) for r in regions)}</div>',
    unsafe_allow_html=True,
)
if len(plan["days"]) < plan["requested_days"]:
    st.warning("Not enough places fit your limits to fill every day. "
               "Try a wider distance, more spending, or switch off “Keep my trip close”.")

tab_plan, tab_map = st.tabs(["Day by day", "Map"])

with tab_plan:
    for dd in plan["days"]:
        d = df[df["day"] == dd["day"]]
        st.markdown(
            f'<div class="tp-day"><div class="tp-day-n">Day {dd["day"]}</div>'
            f'<div class="tp-day-h">{html.escape(dd["hub"])}</div></div>',
            unsafe_allow_html=True,
        )
        if dd["from_hub"]:
            how = "Flight" if dd["transfer_mode"] == "flight" else "Drive"
            st.markdown(
                f'<div class="tp-note">{how} from {html.escape(dd["from_hub"])}: about {dd["transfer_km"]} km, '
                f'roughly {fmt_minutes(dd["transfer_min"])}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="tp-grid">' + "".join(card_html(r) for _, r in d.iterrows()) + "</div>",
                    unsafe_allow_html=True)

with tab_map:
    m = df[["latitude", "longitude", "day"]].copy()
    m["color"] = m["day"].apply(lambda x: DAY_COLORS[(x - 1) % len(DAY_COLORS)])
    st.map(m, latitude="latitude", longitude="longitude", color="color", size=1200)
    legend = "  ".join(
        f'<span style="color:{DAY_COLORS[(d["day"] - 1) % len(DAY_COLORS)]}">●</span> Day {d["day"]}'
        for d in plan["days"]
    )
    st.markdown(f'<div class="tp-meta">{legend}</div>', unsafe_allow_html=True)

c1, c2, _ = st.columns([1, 1, 3])
c1.download_button("Download CSV", df.to_csv(index=False).encode("utf-8-sig"), "itinerary.csv", "text/csv")
c2.download_button("Download JSON", json.dumps(json.loads(df.to_json(orient="records")), ensure_ascii=False, indent=2),
                   "itinerary.json", "application/json")
st.caption("Travel times are estimates. Transport and accommodation costs are not included in the budget.")

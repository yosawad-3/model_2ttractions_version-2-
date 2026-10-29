import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = "places_clean.csv"
SPEED_KMH = 40  # average city travel speed used to estimate travel time

PRICE_COL = {
    ("Egyptian", False): "egyptian_adult_price_egp",
    ("Egyptian", True): "egyptian_student_price_egp",
    ("Arab", False): "arab_adult_price_egp",
    ("Arab", True): "arab_student_price_egp",
    ("Foreigner", False): "foreigner_adult_price_egp",
    ("Foreigner", True): "foreigner_student_price_egp",
}


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_data
def load_places(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tag_list"] = df["tags_str"].fillna("other").apply(lambda s: s.split("|"))
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------
# Model: filtering + scoring + itinerary
# ----------------------------------------------------------------------
def score_places(df, city, selected_tags, nationality, n_adults, n_students):
    """Return places of `city` with an interest score and the group's ticket cost."""
    d = df[df["city_norm"] == city].copy()

    adult_col = PRICE_COL[(nationality, False)]
    student_col = PRICE_COL[(nationality, True)]
    d["group_cost"] = n_adults * d[adult_col] + n_students * d[student_col]

    # content-based part: share of the chosen tags that the place has
    if selected_tags:
        chosen = set(selected_tags)
        d["interest_score"] = d["tag_list"].apply(lambda t: len(chosen & set(t)) / len(chosen))
    else:
        d["interest_score"] = 0.5

    # soft price penalty (log scale, normalised inside the city)
    p = np.log1p(d["group_cost"])
    d["price_penalty"] = p / p.max() if p.max() > 0 else 0.0

    d["final_score"] = d["interest_score"] - 0.15 * d["price_penalty"]
    return d.sort_values("final_score", ascending=False)


def build_itinerary(d, days, places_per_day, daily_hours, total_budget):
    """Greedy: pick best places under the budget, then group them by proximity per day."""
    max_places = days * places_per_day
    daily_minutes = daily_hours * 60

    # 1) hard constraint: total budget
    chosen, spent = [], 0.0
    for idx, row in d.iterrows():
        if len(chosen) >= max_places:
            break
        if spent + row["group_cost"] <= total_budget:
            chosen.append(idx)
            spent += row["group_cost"]
    remaining = d.loc[chosen].copy()

    # 2) split into days: start from best place, then nearest neighbour
    rows = []
    for day in range(1, days + 1):
        if remaining.empty:
            break
        current = remaining.iloc[0]
        order, used = 0, 0.0
        while True:
            travel = 0.0
            if order > 0:
                km = haversine_km(prev["latitude"], prev["longitude"],
                                  current["latitude"], current["longitude"])
                travel = km / SPEED_KMH * 60
            need = travel + current["explore_duration_min"]
            if order > 0 and used + need > daily_minutes:
                break
            order += 1
            used += need
            rows.append({
                "Day": day, "Order": order, "Place": current["place_name"],
                "Category": current["category"],
                "Travel (min)": round(travel),
                "Visit (min)": int(current["explore_duration_min"]),
                "Cost (EGP)": round(current["group_cost"]),
                "latitude": current["latitude"], "longitude": current["longitude"],
                "Image_url": current.get("Image_url"),
            })
            prev = current
            remaining = remaining.drop(current.name)
            if remaining.empty or order >= places_per_day:
                break
            dist = haversine_km(prev["latitude"], prev["longitude"],
                                remaining["latitude"], remaining["longitude"])
            current = remaining.loc[dist.idxmin()]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Egypt Trip Planner", page_icon="🏺", layout="wide")
    st.title("🏺 Egypt Attractions Trip Planner")

    df = load_places()
    cities = sorted(df["city_norm"].dropna().unique())
    all_tags = sorted({t for tags in df["tag_list"] for t in tags if t != "other"})

    with st.sidebar:
        st.header("Trip settings")
        city = st.selectbox("City", cities)
        selected_tags = st.multiselect("Interests (place type)", all_tags)
        days = st.slider("Number of days", 1, 7, 2)
        places_per_day = st.slider("Places per day", 1, 6, 3)
        daily_hours = st.slider("Hours available per day", 2, 12, 8)

        st.subheader("Group")
        nationality = st.selectbox("Ticket category", ["Egyptian", "Arab", "Foreigner"])
        n_adults = st.number_input("Adults", 0, 20, 2)
        n_students = st.number_input("Students", 0, 20, 0)
        total_budget = st.number_input("Total tickets budget (EGP)", 0, 1_000_000, 5000, step=500)

        go = st.button("Plan my trip", type="primary")

    if not go:
        st.info("Choose your settings from the sidebar, then press **Plan my trip**.")
        return
    if n_adults + n_students == 0:
        st.warning("Add at least one traveller.")
        return

    scored = score_places(df, city, selected_tags, nationality, n_adults, n_students)
    plan = build_itinerary(scored, days, places_per_day, daily_hours, total_budget)

    if plan.empty:
        st.error("No places fit your budget in this city. Try a larger budget.")
        return

    st.success(f"{len(plan)} places planned in {city} | "
               f"Total tickets: {plan['Cost (EGP)'].sum():,} EGP")

    for day, g in plan.groupby("Day"):
        st.subheader(f"Day {day}")
        for _, r in g.iterrows():
            c1, c2 = st.columns([1, 3])
            with c1:
                if isinstance(r["Image_url"], str) and r["Image_url"].startswith("http"):
                    st.image(r["Image_url"], width=260)
            with c2:
                st.markdown(f"**{r['Order']}. {r['Place']}**")
                st.caption(r["Category"])
                st.write(f"Visit: {r['Visit (min)']} min | Travel from previous: "
                         f"{r['Travel (min)']} min | Tickets: {r['Cost (EGP)']:,} EGP")

    st.subheader("Map")
    st.map(plan[["latitude", "longitude"]])

    with st.expander("Full table"):
        st.dataframe(plan.drop(columns=["Image_url", "latitude", "longitude"]))


if __name__ == "__main__":
    main()

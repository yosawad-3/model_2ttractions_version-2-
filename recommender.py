"""
Core trip-planning logic (no Streamlit dependency).

The user never picks a city. The planner:
  1. groups nearby places into regions ("hubs") by real distance,
  2. scores places by interests + ticket price for the group,
  3. tries every region as a starting point and keeps the best day-by-day plan.

prefer_nearby=True  -> every region must be within `max_hub_km` of the start region.
prefer_nearby=False -> any region is allowed; travel time (road / flight) is deducted
                       from the day it happens on, and a region is never revisited.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

TICKET_KEYS = [
    "foreigner_adult", "foreigner_student",
    "arab_adult", "arab_student",
    "egyptian_adult", "egyptian_student",
]
REQUIRED_COLUMNS = [
    "place_id", "place_name", "category", "latitude", "longitude",
    "explore_duration_min", "Image_url",
] + [f"{k}_price_egp" for k in TICKET_KEYS]


def haversine_matrix(lat_deg, lon_deg) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, float))[:, None]
    lon = np.radians(np.asarray(lon_deg, float))[:, None]
    a = np.sin((lat - lat.T) / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _hav(lat1, lon1, lat2, lon2) -> float:
    return float(haversine_matrix([lat1, lat2], [lon1, lon2])[0, 1])


class TripPlanner:
    def __init__(
        self,
        df: pd.DataFrame,
        hub_radius_km: float = 50.0,
        road_factor: float = 1.25,
        city_speed_kmh: float = 25.0,
        road_speed_kmh: float = 70.0,
        flight_threshold_km: float = 400.0,
        flight_minutes: int = 240,
        min_visit_min: int = 90,
    ):
        self.road_factor = road_factor
        self.city_speed = city_speed_kmh
        self.road_speed = road_speed_kmh
        self.flight_threshold = flight_threshold_km
        self.flight_minutes = flight_minutes
        self.min_visit_min = min_visit_min

        self.places = self._prepare(df)
        self.D = haversine_matrix(self.places["latitude"], self.places["longitude"])
        self._build_hubs(hub_radius_km)

    # ------------------------------------------------------------------ data
    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "city_norm" not in df.columns:
            if "city" in df.columns:
                df["city_norm"] = df["city"]
            else:
                raise ValueError("Missing column: city_norm (or city)")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in the places file: {missing}")

        # tags may arrive as a real list or as 'a|b|c' text (tags_str)
        if "tags" in df.columns and df["tags"].apply(lambda t: isinstance(t, list)).all():
            pass
        elif "tags_str" in df.columns:
            df["tags"] = df["tags_str"].fillna("other").astype(str).str.split("|")
        else:
            raise ValueError("Missing column: tags_str")

        for c in ["latitude", "longitude", "explore_duration_min"] + [f"{k}_price_egp" for k in TICKET_KEYS]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        for c in ["explore_duration_min"] + [f"{k}_price_egp" for k in TICKET_KEYS]:
            df[c] = df[c].fillna(df[c].median())

        if "is_panorama" in df.columns:
            df = df[df["is_panorama"] == 0]
        df["Image_url"] = df["Image_url"].fillna("").astype(str)
        df["category"] = df["category"].fillna("").astype(str)
        if df.empty:
            raise ValueError("The places file has no usable rows.")
        return df.reset_index(drop=True)

    def _build_hubs(self, radius_km: float) -> None:
        p = self.places
        if len(p) > 1:
            Z = linkage(squareform(self.D, checks=False), method="average")
            p["hub_id"] = fcluster(Z, t=radius_km, criterion="distance")
        else:
            p["hub_id"] = 1
        self.hub_ids = sorted(p["hub_id"].unique())
        self.hubs = {}
        for h in self.hub_ids:
            sub = p[p["hub_id"] == h]
            self.hubs[h] = dict(
                name=str(sub["city_norm"].mode().iat[0]),
                lat=float(sub["latitude"].mean()),
                lon=float(sub["longitude"].mean()),
                n=len(sub),
            )
        seen: dict[str, int] = {}
        for h in self.hub_ids:
            nm = self.hubs[h]["name"]
            seen[nm] = seen.get(nm, 0) + 1
            if seen[nm] > 1:
                self.hubs[h]["name"] = f"{nm} ({seen[nm]})"
        self.hpos = {h: i for i, h in enumerate(self.hub_ids)}
        self.HD = haversine_matrix(
            [self.hubs[h]["lat"] for h in self.hub_ids],
            [self.hubs[h]["lon"] for h in self.hub_ids],
        )
        self.hub_members = {h: list(p.index[p["hub_id"] == h]) for h in self.hub_ids}

    def available_interests(self) -> list[str]:
        counts = self.places["tags"].explode().value_counts()
        return [t for t in counts.index if t != "other"]

    # --------------------------------------------------------------- scoring
    def _scores(self, interests, group, w_interest=0.75, w_price=0.25):
        interests = set(interests or [])
        cost = np.zeros(len(self.places))
        for key, cnt in group.items():
            col = f"{key}_price_egp"
            if col not in self.places.columns:
                raise ValueError(f"Unknown ticket category: {key}")
            cost += cnt * self.places[col].values

        def match(tags):
            if not interests:
                return 0.5
            hits = len(set(tags) & interests)
            return 0.0 if hits == 0 else min(1.0, 0.6 + 0.4 * (hits - 1))

        m = self.places["tags"].apply(match).values
        lc = np.log1p(cost)
        price_norm = lc / max(lc.max(), 1e-9)
        return w_interest * (0.1 + 0.9 * m) + w_price * (1 - price_norm), cost

    # --------------------------------------------------------------- planning
    def transfer_info(self, km: float):
        if km <= 0:
            return 0, "none"
        if km > self.flight_threshold:
            return self.flight_minutes, "flight"
        return int(km * self.road_factor / self.road_speed * 60), "road"

    def _pick(self, pool, avail_min, budget_left, k, score, cost, dur):
        chosen, used_min, spent = [], 0.0, 0.0
        for i in sorted(pool, key=lambda j: -score[j]):
            if len(chosen) >= k:
                break
            if cost[i] > budget_left - spent:
                continue
            extra = 0.0
            if chosen:
                extra = min(self.D[i, c] for c in chosen) * self.road_factor / self.city_speed * 60
            need = dur[i] + extra
            if used_min + need > avail_min:
                continue
            chosen.append(i)
            used_min += need
            spent += cost[i]
        return chosen, used_min, spent

    def _order_path(self, chosen, lat, lon):
        left, order, cur = list(chosen), [], (lat, lon)
        P = self.places
        while left:
            nxt = min(left, key=lambda i: _hav(cur[0], cur[1], P.at[i, "latitude"], P.at[i, "longitude"]))
            order.append(nxt)
            left.remove(nxt)
            cur = (P.at[nxt, "latitude"], P.at[nxt, "longitude"])
        return order

    def _run(self, start_h, days, k, budget, daily_min, nearby, max_hub_km, score, cost, dur):
        lam = 0.5 if nearby else 0.1               # travel penalty per 100 km
        used, visited, out_days = set(), [], []
        prev_h, budget_left, total = None, budget, 0.0

        for d in range(1, days + 1):
            best = None
            for h in self.hub_ids:
                if prev_h is None and h != start_h:
                    continue
                if nearby and self.HD[self.hpos[start_h], self.hpos[h]] > max_hub_km:
                    continue
                if h != prev_h and h in visited:
                    continue
                if prev_h is not None and h != prev_h:
                    km = self.HD[self.hpos[prev_h], self.hpos[h]]
                    if nearby and km > max_hub_km:
                        continue
                    t_min, t_mode = self.transfer_info(km)
                    avail, pen = daily_min - t_min, lam * km / 100
                else:
                    km, t_min, t_mode, avail, pen = 0.0, 0, "none", daily_min, 0.0
                if avail < self.min_visit_min:
                    continue
                pool = [i for i in self.hub_members[h] if i not in used]
                day_cap = min(budget_left, budget_left / (days - d + 1) * 1.5)
                picked, used_min, spent = self._pick(pool, avail, day_cap, k, score, cost, dur)
                if not picked:
                    continue
                gain = float(sum(score[i] for i in picked)) - pen
                if best is None or gain > best["gain"]:
                    best = dict(h=h, picked=picked, gain=gain, km=km, t_min=t_min,
                                t_mode=t_mode, used_min=used_min, spent=spent)
            if best is None:
                break
            h = best["h"]
            order = self._order_path(best["picked"], self.hubs[h]["lat"], self.hubs[h]["lon"])
            out_days.append(dict(
                day=d, hub_id=h, hub=self.hubs[h]["name"], order=order,
                from_hub=self.hubs[prev_h]["name"] if (prev_h is not None and prev_h != h) else None,
                transfer_km=round(best["km"]), transfer_min=best["t_min"], transfer_mode=best["t_mode"],
                minutes=int(best["used_min"]), cost=float(best["spent"]),
            ))
            used.update(best["picked"])
            budget_left -= best["spent"]
            total += best["gain"]
            if h != prev_h:
                visited.append(h)
            prev_h = h
        return dict(start=start_h, days=out_days, total=total)

    def plan(self, days=5, places_per_day=3, budget_egp=None, group=None, interests=None,
             prefer_nearby=True, max_hub_km=150, daily_hours=8) -> dict:
        group = {k: v for k, v in (group or {"foreigner_adult": 1}).items() if v > 0}
        if not group:
            raise ValueError("Add at least one traveller.")
        budget = float("inf") if budget_egp is None else float(budget_egp)
        score, cost = self._scores(interests, group)
        dur = self.places["explore_duration_min"].values.astype(float)

        best = None
        for h0 in self.hub_ids:
            p = self._run(h0, days, places_per_day, budget, daily_hours * 60,
                          prefer_nearby, max_hub_km, score, cost, dur)
            if p["days"] and (best is None or p["total"] > best["total"]):
                best = p
        if best is None:
            raise RuntimeError("No plan fits these limits. Try a larger budget or more hours per day.")

        rows = []
        for dd in best["days"]:
            for o, i in enumerate(dd["order"], 1):
                r = self.places.loc[i]
                rows.append(dict(
                    day=dd["day"], order=o, hub=dd["hub"], place_id=r["place_id"],
                    place_name=r["place_name"], city=r["city_norm"], category=r["category"],
                    duration_min=int(r["explore_duration_min"]), ticket_cost_egp=float(cost[i]),
                    tags=", ".join(r["tags"]),
                    matched=", ".join(sorted(set(r["tags"]) & set(interests or []))),
                    latitude=float(r["latitude"]), longitude=float(r["longitude"]),
                    Image_url=r["Image_url"],
                ))
        seq = [d["hub_id"] for d in best["days"]]
        hub_km = sum(self.HD[self.hpos[a], self.hpos[b]] for a, b in zip(seq, seq[1:]) if a != b)
        return dict(
            itinerary=pd.DataFrame(rows), days=best["days"],
            requested_days=days, prefer_nearby=prefer_nearby, max_hub_km=max_hub_km,
            total_ticket_cost=sum(d["cost"] for d in best["days"]),
            budget=None if budget_egp is None else float(budget_egp),
            hub_km=float(hub_km),
        )

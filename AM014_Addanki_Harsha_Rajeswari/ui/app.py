"""PathSnap evaluator UI (runs on laptop 2).

Start:
    streamlit run AM014_Addanki_Harsha_Rajeswari/ui/app.py -- --api http://<laptop-1-ip>:8000

The UI only talks to the API over HTTP. It never reads trip files, model files
or reference routes.
"""
import argparse
import math
import os

import folium
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


def default_api():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000"))
    args, _ = ap.parse_known_args()
    return args.api


st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching evaluation")

with st.sidebar:
    api = st.text_input("API address", value=default_api()).rstrip("/")
    try:
        h = requests.get(f"{api}/health", timeout=5).json()
        st.success(f"API reachable - model: {h.get('model')}")
        st.caption(f"trips available: {h.get('n_trips')}")
    except Exception as e:  # noqa: BLE001
        st.error(f"API not reachable: {e}")
        st.stop()


@st.cache_data(ttl=60)
def get_trips(api_url):
    return requests.get(f"{api_url}/trips", timeout=10).json()["trips"]


trips = get_trips(api)
labels = {f"{t['trip_id']}  ({t['n_fixes']} GPS fixes)": t["trip_id"] for t in trips}
choice = st.radio("Evaluation trips", list(labels), horizontal=True)
run = st.button("Run map matching", type="primary")

if "history" not in st.session_state:
    st.session_state.history = []

if run:
    trip_id = labels[choice]
    with st.spinner(f"Sending trip {trip_id} to {api} ..."):
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state.result = r.json()
    res = st.session_state.result
    m = res.get("metrics") or {}
    st.session_state.history.append(dict(
        trip_id=res["trip_id"], fixes=res["n_fixes"],
        top1_hmm=m.get("top1_accuracy_hmm"), top1_classifier=m.get("top1_accuracy_classifier"),
        continuity=m.get("route_continuity_hmm"), route_f1=m.get("route_f1"),
        seconds=res["timings_s"]["request_total_s"]))

res = st.session_state.get("result")
if res:
    m = res.get("metrics")
    st.subheader(f"Trip {res['trip_id']} - {res['n_fixes']} GPS fixes - model: {res['model']}")
    if m:
        c = st.columns(3)
        c[0].metric("Top-1 accuracy (HMM)", f"{m['top1_accuracy_hmm']:.2%}",
                    f"{m['top1_accuracy_hmm'] - m['top1_accuracy_classifier']:+.2%} vs classifier only")
        c[1].metric("Top-1 accuracy (classifier only)", f"{m['top1_accuracy_classifier']:.2%}")
        c[2].metric("Route continuity", f"{m['route_continuity_hmm']:.2%}")
        c = st.columns(3)
        c[0].metric("Route F1 (length-weighted)", f"{m['route_f1']:.2%}")
        c[1].metric("Labelled fixes scored", f"{m['n_labelled_fixes']} / {m['n_fixes']}")
        c[2].metric("API time", f"{res['timings_s']['request_total_s']:.2f} s")
        with st.expander("All metrics returned by the API"):
            st.json(m)
            st.json(res["timings_s"])
    else:
        st.info("No reference route on the server for this trip; metrics unavailable.")

    lat = [p[0] for p in res["gps"]]
    lon = [p[1] for p in res["gps"]]
    # centre and zoom from the track extent (fit_bounds is unreliable inside the embedded iframe)
    span = max(max(lat) - min(lat), (max(lon) - min(lon)) * 0.7, 1e-4)
    # at zoom z the world is 256 * 2**z px wide; fit the track into ~550 px
    zoom = int(max(3, min(17, math.floor(math.log2(360 * 550 / (256 * span))))))
    # Esri tiles: the OpenStreetMap tile server rejects requests without a Referer,
    # which is what an embedded srcdoc iframe sends
    fmap = folium.Map(location=[(min(lat) + max(lat)) / 2, (min(lon) + max(lon)) / 2], zoom_start=zoom,
                      tiles="Esri.WorldStreetMap", control_scale=True, height="600px")
    fg = folium.FeatureGroup(name="Candidate road segments")
    folium.PolyLine(res["candidate_network"], color="#9a9a9a", weight=2, opacity=0.6).add_to(fg)
    fg.add_to(fmap)
    fg = folium.FeatureGroup(name="GPS observations")
    for p in res["gps"]:
        folium.CircleMarker(p, radius=2, color="#1f4fff", fill=True, weight=0, fill_opacity=0.6).add_to(fg)
    fg.add_to(fmap)
    fg = folium.FeatureGroup(name="Predicted route")
    for line in res["predicted_route"]:
        folium.PolyLine(line, color="#d7301f", weight=3, opacity=0.9, tooltip="predicted route").add_to(fg)
    fg.add_to(fmap)
    folium.Marker(res["gps"][0], tooltip="start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(res["gps"][-1], tooltip="end", icon=folium.Icon(color="black")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st.markdown("**Map** - blue: GPS observations, red: predicted route, grey: candidate road segments")
    components.html(fmap.get_root().render(), height=620)

if st.session_state.history:
    st.subheader("Requests in this session")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

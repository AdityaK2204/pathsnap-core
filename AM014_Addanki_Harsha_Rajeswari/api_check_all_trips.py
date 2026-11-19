"""Send every trip served by the API through /predict and print a summary table.

Usage (API must be running):
    python -m AM014_Addanki_Harsha_Rajeswari.api_check_all_trips --api http://127.0.0.1:8000
"""
import argparse

import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=None, help="optional CSV path")
    args = ap.parse_args()
    api = args.api.rstrip("/")
    trips = requests.get(f"{api}/trips", timeout=30).json()["trips"]
    rows = []
    for t in trips:
        r = requests.post(f"{api}/predict", json={"trip_id": t["trip_id"]}, timeout=600)
        r.raise_for_status()
        res = r.json()
        m = res.get("metrics") or {}
        rows.append(dict(trip_id=res["trip_id"], fixes=res["n_fixes"],
                         top1_hmm=m.get("top1_accuracy_hmm"), top1_classifier=m.get("top1_accuracy_classifier"),
                         continuity_hmm=m.get("route_continuity_hmm"), route_f1=m.get("route_f1"),
                         seconds=res["timings_s"]["request_total_s"]))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

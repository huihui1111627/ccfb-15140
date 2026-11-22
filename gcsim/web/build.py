"""Build the offline HTML dashboard from an acceptance JSON document.

The dashboard is a single self-contained file: the data document is embedded
inline, so it can be opened from disk with no server and no network access.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "template.html")
PLACEHOLDER = "__DATA__"


def build_dashboard(acceptance: dict, out_path: str) -> str:
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        template = f.read()
    # one population per scenario (the dashboard shows the current one's
    # lifetime curves); use the first scenario as the default
    populations = {}
    if acceptance.get("scenarios"):
        pop = acceptance["scenarios"][0].get("population") or {}
        populations = pop
    payload = dict(acceptance)
    payload["populations"] = populations
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # guard against a literal "</script>" inside the JSON
    data = data.replace("</", "<\\/")
    html = template.replace(PLACEHOLDER, data)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def load_and_build(json_path: str, out_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        acceptance = json.load(f)
    return build_dashboard(acceptance, out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the offline dashboard")
    ap.add_argument("--input", default="reports/acceptance.json")
    ap.add_argument("--out", default="reports/dashboard.html")
    args = ap.parse_args()
    path = load_and_build(args.input, args.out)
    size = os.path.getsize(path)
    print(f"wrote {path} ({size / 1024:.0f} KiB)")

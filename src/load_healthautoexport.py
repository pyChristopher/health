import json
from pathlib import Path
from typing import Any, Dict, List

def load_payloads(data_dir: str = "data") -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for p in sorted(Path(data_dir).glob("HealthAutoExport-*.json")):
        with open(p, "r", encoding="utf-8") as f:
            payloads.append(json.load(f))
    return payloads

def get_root(payload: Dict[str, Any]) -> Dict[str, Any]:
    # HealthAutoExport typically nests under "data"
    return payload.get("data", payload)

def iter_metrics(payload: Dict[str, Any]):
    root = get_root(payload)
    for m in root.get("metrics", []):
        yield m

def iter_workouts(payload: Dict[str, Any]):
    root = get_root(payload)
    for w in root.get("workouts", []):
        yield w

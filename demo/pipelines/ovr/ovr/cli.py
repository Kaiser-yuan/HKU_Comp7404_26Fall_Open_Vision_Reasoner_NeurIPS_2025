import argparse
import json
import os
import urllib.request
from .common import ROOT, MODELS, fingerprint, read_json, read_jsonl, write_json


def access_check():
    cfg = read_json(ROOT / "configs/protocol.json")
    result = []
    for name, spec in cfg["models"].items():
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/config.json"
        headers = {}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers, method="HEAD"), timeout=30) as r:
                status = r.status
        except urllib.error.HTTPError as e:
            status = e.code
        result.append({"model": name, "http_status": status, "config_accessible": status == 200,
                       "weight_access": "NOT CHECKED - verify before leasing GPU"})
    write_json(ROOT / "results/reports/model_access.json", result)
    return result


def estimate(stage, rate, setup, reserve):
    if rate <= 0 or setup < 0 or reserve < 0:
        raise ValueError("Rate must be positive; setup/reserve must be nonnegative")
    models = []
    for model in MODELS:
        paths = list((ROOT / "results/raw/pilot").glob(f"{model}.session-*.json"))
        sessions = [read_json(p) for p in paths]
        seconds = sum(s.get("elapsed_s", 0) for s in sessions)
        n = sum(s.get("completed_this_session", 0) for s in sessions)
        if not n:
            models.append({"model": model, "state": "NEEDS PILOT - no invented throughput"})
        else:
            hours = seconds / n * 3040 / 3600
            models.append({"model": model, "pilot_n": n, "projected_hours": hours,
                           "projected_cny": hours * rate, "conservative_hours": hours * 1.5})
    ready = all("projected_hours" in m for m in models)
    result = {"status": "ESTIMATE ONLY - NOT APPROVAL", "hourly_rate_cny": rate,
              "models": models, "includes_pilot": False,
              "note": "Measured session time includes model startup; 50% runtime contingency. Provider charges and shutdown must be controlled externally."}
    if ready:
        result["proposed_full_budget_cny"] = (sum(m["conservative_hours"] for m in models) + setup) * rate + reserve
    write_json(ROOT / "results/reports/budget_estimate.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description="OVR course project tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["prepare-data", "validate-data", "access-check", "fingerprint", "annotations", "analyze", "build-demo", "freeze"]:
        sub.add_parser(name)
    runp = sub.add_parser("run")
    runp.add_argument("--model", choices=MODELS, required=True)
    runp.add_argument("--stage", choices=["pilot", "full"], required=True)
    runp.add_argument("--approval", required=True)
    scorep = sub.add_parser("score")
    scorep.add_argument("--model", choices=MODELS, required=True)
    scorep.add_argument("--stage", choices=["pilot", "full"], required=True)
    reportp = sub.add_parser("report")
    reportp.add_argument("--stage", choices=["pilot", "full"], default="full")
    ep = sub.add_parser("estimate")
    ep.add_argument("--hourly-rate", type=float, required=True)
    ep.add_argument("--setup-hours", type=float, default=1)
    ep.add_argument("--storage-reserve", type=float, default=0)
    args = p.parse_args()
    if args.cmd == "prepare-data":
        from .data import prepare
        result = prepare()
    elif args.cmd == "validate-data":
        from .data import validate
        result = validate()
    elif args.cmd == "access-check":
        result = access_check()
    elif args.cmd == "fingerprint":
        result = {"protocol_sha256": fingerprint(read_json(ROOT / "configs/protocol.json"))}
    elif args.cmd == "run":
        from .run import run
        result = run(args.model, args.stage, args.approval)
    elif args.cmd == "score":
        from .scoring import score
        result = score(ROOT / f"results/raw/{args.stage}/{args.model}.jsonl", ROOT / "data/questions.jsonl",
                       ROOT / f"results/scored/{args.stage}/{args.model}.jsonl")
    elif args.cmd == "report":
        from .report import build_report
        result = build_report(args.stage)
        # Full missing-ID lists are saved, not printed.
        result = {k: v for k, v in result.items() if k != "missing_ids"}
    elif args.cmd == "freeze":
        from .report import freeze
        result = freeze()
    elif args.cmd == "estimate":
        result = estimate("pilot", args.hourly_rate, args.setup_hours, args.storage_reserve)
    elif args.cmd == "annotations":
        from .analysis import annotation_template
        result = annotation_template()
    elif args.cmd == "analyze":
        from .analysis import analyze
        result = analyze()
    elif args.cmd == "build-demo":
        from .analysis import build_demo
        result = build_demo()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

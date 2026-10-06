"""LLM call ledger for the typed-pipeline work (2026-10-03; the user: "if you need LLM calls, limit it to 200 calls as
for now"). Every script reserves its calls here before making them; reserve() refuses past the cap."""
import json, os, time
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_ledger.json")
CAP = 200


def used() -> int:
    return sum(r["calls"] for r in json.load(open(PATH))) if os.path.exists(PATH) else 0


def reserve(name: str, calls: int) -> None:
    if used() + calls > CAP:
        raise SystemExit(f"LLM cap: {used()} used + {calls} asked > {CAP}")
    rows = json.load(open(PATH)) if os.path.exists(PATH) else []
    rows.append({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "calls": calls})
    json.dump(rows, open(PATH, "w"), indent=1)


def pending(name: str, keys) -> int:
    """How many of these keys reader_net/llm.run would still call for (its cache: llm/<name>.jsonl)."""
    path = f"/root/zadumai_nli_proto/reader_net/llm/{name}.jsonl"
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    return sum(k not in done for k in keys)

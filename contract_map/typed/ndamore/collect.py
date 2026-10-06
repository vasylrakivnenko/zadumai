"""More NDAs for the CNN (2026-10-04; the user chose step 2: more training data, labeled by our 92% system). Collects
stand-alone NDAs filed as exhibits on SEC EDGAR (full-text search, 2001-), with the user's approved contact in the
User-Agent (sec.gov only; memory: sec-edgar-access), at most ~8 requests / s (SEC's limit is 10).
  1. search  NDA phrasings x filing types, exhibits only (file_type EX-...), paginated
  2. fetch   each exhibit (HTML -> text), resumable
  3. filter  an NDA title near the top (confidentiality / non-disclosure / secrecy agreement or letter), not a merger,
             employment, purchase or separation agreement, 2,000-80,000 characters
  4. dedupe  against EVERY ContractNLI NDA (train, dev, test: the sealed sets too) and among themselves: 8-word shingles,
             containment >= 0.3 -> dropped (the same NDA is often filed twice, e.g. with the SC TO-T and the SC 14D9)
usage: python collect.py search|search2|fetch|filter   -> data/hits.jsonl, data/raw.jsonl, data/ndas.jsonl"""
import html, json, os, re, sys, threading, time, urllib.parse, urllib.request, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/data", exist_ok=True)
UA = "Zadum research " + next((l.split("=", 1)[1].strip() for l in open("/root/.env") if l.startswith("SEC_CONTACT_EMAIL=")), "")  # SEC asks for a contact email (in /root/.env)
QUERIES = ['"Evaluation Material" "Confidentiality Agreement"', '"non-disclosure agreement" "Receiving Party" "Disclosing Party"',
           '"nondisclosure agreement" "Confidential Information" "Recipient"', '"confidentiality agreement" "Receiving Party" "Disclosing Party"',
           '"mutual nondisclosure agreement"', '"mutual non-disclosure agreement"', '"confidentiality agreement" "Representatives" "Evaluation Material"',
           '"non-disclosure agreement" "Confidential Information" "Recipient"', '"secrecy agreement" "Confidential Information"']
FORMS = ["SC TO-T,SC TO-C,SC 14D9,SC 13E3,SC 13D", "8-K,8-K/A", "DEFM14A,PREM14A,DEFA14A,S-4,S-4/A", "10-K,10-Q,S-1,F-4,6-K"]
PER_QUERY = 2000
_lock = threading.Lock(); _last = [0.0]


def get(url):
    with _lock:  # <= ~8 requests / s across threads
        wait = 0.125 - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
    for i in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"}), timeout=60).read()
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(2 * (i + 1))
    raise err


def search():
    out, seen = open(f"{HERE}/data/hits.jsonl", "a"), set()
    if os.path.exists(f"{HERE}/data/hits.jsonl"):
        seen = {json.loads(l)["id"] for l in open(f"{HERE}/data/hits.jsonl")}
    for q in QUERIES:
        for forms in FORMS:
            start = 0
            while start < PER_QUERY:
                url = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode({"q": q, "forms": forms, "from": start})
                try: d = json.loads(get(url))
                except Exception as e: print("search failed", q, forms, start, e, flush=True); break  # noqa: BLE001
                hits = d.get("hits", {}).get("hits", [])
                for h in hits:
                    s = h["_source"]
                    if not (s.get("file_type") or "").upper().startswith("EX-") or h["_id"] in seen: continue
                    seen.add(h["_id"])
                    out.write(json.dumps({"id": h["_id"], "cik": int(s["ciks"][0]), "form": s.get("form"), "date": s.get("file_date"),
                                          "type": s.get("file_type"), "desc": s.get("file_description"), "q": q}) + "\n")
                out.flush()
                total = d.get("hits", {}).get("total", {}).get("value", 0)
                start += 100
                if not hits or start >= total: break
            print(f"{q} | {forms}: {len(seen)} exhibits so far", flush=True)


QUERIES2 = ['"Evaluation Material" "Representatives"', '"confidentiality agreement" "standstill"',
            '"Confidential Information" "Receiving Party" "return or destroy"', '"non-disclosure agreement"',
            '"confidentiality agreement" "Disclosing Party"', '"nondisclosure agreement" "Disclosing Party"',
            '"proprietary information" "Recipient" "non-disclosure"', '"Evaluation Material" "Confidentiality Agreement"']
FORMS2 = ["SC TO-T,SC TO-T/A,SC TO-C,SC 14D9,SC 14D9/A,SC 13E3,SC 13E3/A,SC 13D,SC 13D/A", "8-K,8-K/A,6-K",
          "DEFM14A,PREM14A,DEFA14A,DEFM14C,S-4,S-4/A,F-4,F-4/A,425", "10-K,10-K/A,10-Q,10-Q/A,S-1,S-1/A,F-1,20-F,40-F"]


def search2():
    """Wider search (2026-10-04, the user: more SEC NDAs): QUERIES2 x FORMS2 x each year 2001-2026 (the engine returns at
    most 10,000 hits per query, so years keep each query under the cap); appends new exhibits to hits.jsonl."""
    seen = {json.loads(l)["id"] for l in open(f"{HERE}/data/hits.jsonl")} if os.path.exists(f"{HERE}/data/hits.jsonl") else set()
    n0 = len(seen)
    with open(f"{HERE}/data/hits.jsonl", "a") as out:
        for q in QUERIES2:
            for forms in FORMS2:
                for year in range(2001, 2027):
                    start = 0
                    while start < 10000:
                        url = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode(
                            {"q": q, "forms": forms, "dateRange": "custom", "startdt": f"{year}-01-01", "enddt": f"{year}-12-31", "from": start})
                        try: d = json.loads(get(url))
                        except Exception as e: print("search failed", q, forms, year, start, e, flush=True); break  # noqa: BLE001
                        hits = d.get("hits", {}).get("hits", [])
                        for h in hits:
                            s_ = h["_source"]
                            if not (s_.get("file_type") or "").upper().startswith("EX-") or h["_id"] in seen: continue
                            seen.add(h["_id"])
                            out.write(json.dumps({"id": h["_id"], "cik": int(s_["ciks"][0]), "form": s_.get("form"), "date": s_.get("file_date"),
                                                  "type": s_.get("file_type"), "desc": s_.get("file_description"), "q": q}) + "\n")
                        out.flush()
                        total = d.get("hits", {}).get("total", {}).get("value", 0); start += 100
                        if not hits or start >= total: break
                print(f"{q} | {forms[:20]}: {len(seen) - n0} new exhibits so far", flush=True)


def to_text(raw):
    t = raw.decode("utf-8", "ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"[ \t\xa0]+", " ", t); t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def fetch():
    from concurrent.futures import ThreadPoolExecutor
    hits = [json.loads(l) for l in open(f"{HERE}/data/hits.jsonl")]
    if os.path.exists(f"{HERE}/data/fetch_ids.json"):  # NDA-named or generically described exhibits only (7,870 of 11,477)
        keep = set(json.load(open(f"{HERE}/data/fetch_ids.json"))); hits = [h for h in hits if h["id"] in keep]
    done = {json.loads(l)["id"] for l in open(f"{HERE}/data/raw.jsonl")} if os.path.exists(f"{HERE}/data/raw.jsonl") else set()
    todo = [h for h in hits if h["id"] not in done]
    print(f"{len(todo)} exhibits to fetch ({len(done)} done)", flush=True)

    def one(h):
        adsh, fn = h["id"].split(":", 1)
        url = f"https://www.sec.gov/Archives/edgar/data/{h['cik']}/{adsh.replace('-', '')}/{fn}"
        try: return {**h, "url": url, "text": to_text(get(url))[:120000]}
        except Exception as e: return {**h, "url": url, "error": str(e)[:200]}  # noqa: BLE001
    n = 0
    with ThreadPoolExecutor(4) as ex, open(f"{HERE}/data/raw.jsonl", "a") as f:
        for r in ex.map(one, todo):
            f.write(json.dumps(r) + "\n"); n += 1
            if n % 250 == 0: print(f"fetched {n}/{len(todo)}", flush=True)
    print("fetched", n)


TITLE = re.compile(r"\b(confidential(?:ity)?|non-?\s?disclosure|nondisclosure|secrecy|proprietary information)\b[^\n]{0,40}\b(agreement|letter|undertaking)\b", re.I)
NOT_NDA = re.compile(r"\b(merger|acquisition agreement|purchase agreement|employment|separation|severance|consulting agreement|"
                     r"stock option|credit agreement|loan|lease|license agreement|supply agreement|settlement)\b", re.I)


def shingles(t, n=8):
    w = re.findall(r"[a-z0-9]+", t.lower())
    return {hash(" ".join(w[i:i + n])) for i in range(0, max(len(w) - n + 1, 0))}


def filt():
    z = zipfile.ZipFile(f"{HERE}/../../data/ext/contractnli.zip")
    ref = []
    for sp in ("train", "dev", "test"):
        ref += [shingles(d["text"]) for d in json.loads(z.read(f"contract-nli/{sp}.json"))["documents"]]
    index = {}
    for k, s in enumerate(ref):
        for x in list(s)[::7]: index.setdefault(x, []).append(k)  # a sample of each reference NDA's shingles
    stats = {"fetched": 0, "error": 0, "no title": 0, "not an NDA": 0, "length": 0, "ContractNLI overlap": 0, "duplicate": 0, "kept": 0}
    kept, kept_sh, kept_index = [], [], {}
    for l in open(f"{HERE}/data/raw.jsonl"):
        r = json.loads(l); stats["fetched"] += 1
        if "error" in r: stats["error"] += 1; continue
        t = r["text"]; head = t[:3000]  # letter-form NDAs ("Ladies and Gentlemen: In connection with your consideration of
        m = TITLE.search(head)           # a possible transaction ... Evaluation Material") have no title: accepted too;
        em = re.search(r"evaluation material", head, re.I)  # Jev's "is this an NDA?" (label.py) is the last gate
        if not m and not em: stats["no title"] += 1; continue
        cut = min(m.end() + 80 if m else 1500, 1500)
        if NOT_NDA.search(t[:cut]) or NOT_NDA.search(r.get("desc") or ""): stats["not an NDA"] += 1; continue
        if not 2000 <= len(t) <= 80000: stats["length"] += 1; continue
        sh = shingles(t)
        if not sh: stats["length"] += 1; continue
        cand = {k for x in sh if x in index for k in index[x]}
        if any(len(sh & ref[k]) / min(len(sh), len(ref[k])) >= 0.3 for k in cand): stats["ContractNLI overlap"] += 1; continue
        near = {k for x in sh if x in kept_index for k in kept_index[x]}
        if any(len(sh & kept_sh[k]) / min(len(sh), len(kept_sh[k])) >= 0.3 for k in near): stats["duplicate"] += 1; continue
        for x in list(sh)[::7]: kept_index.setdefault(x, []).append(len(kept_sh))
        kept.append({k: r[k] for k in ("id", "url", "form", "date", "type", "desc")} | {"text": t}); kept_sh.append(sh); stats["kept"] += 1
    with open(f"{HERE}/data/ndas.jsonl", "w") as f:
        for r in kept: f.write(json.dumps(r) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    {"search": search, "search2": search2, "fetch": fetch, "filter": filt}[sys.argv[1]]()

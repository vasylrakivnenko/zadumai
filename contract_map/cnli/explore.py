"""Read the training NDAs' gold evidence for one question (train only; dev and test are not read here).
usage: explore.py nda-11 [--n 12] [--grep REGEX]"""
import argparse, collections, json, random, re, zipfile

ZIP = "/root/zadumai_nli_proto/contract_map/data/ext/contractnli.zip"


def docs(split="train"):
    return json.loads(zipfile.ZipFile(ZIP).read(f"contract-nli/{split}.json"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("q"); ap.add_argument("--n", type=int, default=12); ap.add_argument("--grep")
    a = ap.parse_args()
    d = docs(); lab = d["labels"][a.q]
    print(f"{a.q}: {lab['short_description']} | {lab['hypothesis']}")
    by = collections.defaultdict(list); cnt = collections.Counter()
    for doc in d["documents"]:
        an = doc["annotation_sets"][0]["annotations"][a.q]; cnt[an["choice"]] += 1
        for i in an["spans"]:
            s, e = doc["spans"][i]; by[an["choice"]].append(re.sub(r"\s+", " ", doc["text"][s:e]).strip())
    print("labels:", dict(cnt))
    for ch in ("Entailment", "Contradiction"):
        xs = by[ch]; random.Random(0).shuffle(xs)
        print(f"\n-- {ch}: {len(xs)} evidence spans; e.g.")
        for x in xs[:a.n]: print("  *", x[:260])
    if a.grep:
        rx = re.compile(a.grep, re.I); hits = collections.Counter()
        for doc in d["documents"]:
            ch = doc["annotation_sets"][0]["annotations"][a.q]["choice"]
            hits[(ch, bool(rx.search(doc["text"])))] += 1
        print(f"\n/{a.grep}/ in the whole NDA, by label:", dict(hits))
        shown = 0
        for doc in d["documents"]:
            an = doc["annotation_sets"][0]["annotations"][a.q]
            if an["choice"] == "NotMentioned" and (m := rx.search(doc["text"])) and shown < 6:
                print("  N but matches:", re.sub(r"\s+", " ", doc["text"][max(0, m.start() - 150):m.end() + 150])); shown += 1


if __name__ == "__main__":
    main()

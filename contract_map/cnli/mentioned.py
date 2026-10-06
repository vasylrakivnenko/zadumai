"""Can broad topic words tell "this NDA says nothing about the question" (2026-10-03)? For each question: if no
sentence of the whole NDA matches the question's topic words, answer "not mentioned". Shown per question on the
training NDAs (tuned) and dev (check): how many not-mentioned NDAs it would answer, and how many of those calls
would be wrong (the NDA does address it).
usage: mentioned.py [train|dev]"""
import collections, json, re, sys, zipfile

ZIP = "/root/zadumai_nli_proto/contract_map/data/ext/contractnli.zip"
TOPIC = {  # broad on purpose: a match means "maybe mentioned"; only no match at all answers "not mentioned"
    "nda-1": r"\bmark(?:ed|ing|s)?\b|\blabel(?:l)?ed\b|\bdesignat|\bidentif(?:y|ied|ies|ication)\b|\blegend|\bstamp",
    "nda-2": r"\bconfidential information\b|\bproprietary information\b|\bevaluation material|\binformation\b.{0,40}\b(?:means|shall mean|include)",
    "nda-3": r"\boral|\bverbal|\bvisual|\bin any form|\bany (?:form|medium|manner)|\bwhether (?:written|in writing|oral|tangible)|\bor otherwise\b|\bobservation|\binspection|\bin writing\b",
    "nda-4": r"\b(?:use|used|using|utilize)\b",
    "nda-5": r"\bemployee|\bstaff\b|\bpersonnel\b|\bofficers?\b|\bdirectors?\b|\brepresentatives?\b|\bneed to know|\bneed-to-know",
    "nda-7": r"\bconsultant|\badvis[oe]r|\bagents?\b|\bcontractor|\battorney|\baccountant|\bcounsel|\bthird[- ]part|\brepresentatives?\b|\baffiliate|\blenders?\b|\bfinancing sources",
    "nda-8": r"\brequired by|\brequire[sd]? (?:by|to)|\bsubpoena|\bcourt|\blegal(?:ly)?\b|\bcompelled|\bcompulsion|\bgovernmental|\bjudicial|\blaw\b|\blaws\b|\bregulat|\bauthorit",
    "nda-10": r"\bexistence\b|\bterms of (?:this|the) (?:agreement|letter)|\bfact that\b|\bnature of|\bdiscussions|\bnegotiations|\bannounce|\bpublicity|\bpress release|\btransaction is|\bconsidering|\bcontents of this",
    "nda-11": r"reverse[- ]?engineer|decompil|disassembl|source code|\banaly[sz]e\b|\bdecompos|\bderive\b|\bdeconstruct|\bchemical analysis",
    "nda-12": r"\bindependent(?:ly)?\b|\bwithout (?:use of|reference to|access to|the use)",
    "nda-13": r"third[- ]part|\bfrom (?:a|any|another) (?:source|person|third|party)|\blawfully|\brightfully|\bwithout restriction|\bsource other than|\bnot under (?:an )?obligation|\bnon-confidential basis",
    "nda-15": r"\blicen[cs]|\bright|\btitle\b|\bownership|\bproperty of|\bgrant",
    "nda-16": r"\breturn|\bdestroy|\bdestruction|\bdelete|\berase|\bexpunge|\bpurge",
    "nda-17": r"\bcop(?:y|ies|ying|ied)\b|\breproduc|\bduplicat|\breplicat",
    "nda-18": r"\bsolicit|\bhire\b|\bhiring\b|\bemploy\b|\bemployment\b|\brecruit|\bpoach|\bentice|\binduce",
    "nda-19": r"\bsurviv|\bcontinue|\bremain in|\bafter (?:the )?(?:termination|expiration|date)|\bnotwithstanding|\bperiod of|\byears?\b|\bterminat|\bexpir|\bin perpetuity|\bindefinite",
    "nda-20": r"\bretain|\bkeep\b|\barchiv|\bback-?up|\bone copy|\bfor (?:its|their) (?:records|files)|\blegal (?:department|counsel)|\bcompliance purposes|\bautomatic",
}
RX = {k: re.compile(v, re.I) for k, v in TOPIC.items()}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    d = json.loads(zipfile.ZipFile(ZIP).read(f"contract-nli/{split}.json"))
    st = collections.defaultdict(collections.Counter)
    for doc in d["documents"]:
        t = doc["text"]
        for k, rx in RX.items():
            g = doc["annotation_sets"][0]["annotations"][k]["choice"]; hit = bool(rx.search(t))
            st[k]["N" if g == "NotMentioned" else "M"] += 1
            if not hit: st[k]["call_" + ("ok" if g == "NotMentioned" else "wrong")] += 1
    print(f"== {split}: {len(d['documents'])} NDAs. 'not mentioned' when no topic word appears anywhere:")
    tot = collections.Counter()
    for k, c in st.items():
        tot.update(c); calls = c["call_ok"] + c["call_wrong"]
        print(f"  {k:6} not mentioned {c['N']:3} | answered {c['call_ok']:3} ({c['call_ok'] / max(c['N'], 1):4.0%} of them) | wrong calls {c['call_wrong']}"
              + (f" | precision {c['call_ok'] / calls:.1%}" if calls else ""))
    calls = tot["call_ok"] + tot["call_wrong"]
    print(f"  all    not mentioned {tot['N']} | answered {tot['call_ok']} ({tot['call_ok'] / tot['N']:.0%}) | wrong {tot['call_wrong']} | precision {tot['call_ok'] / max(calls, 1):.1%}")


if __name__ == "__main__":
    main()

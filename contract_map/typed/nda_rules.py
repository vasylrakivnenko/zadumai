"""Symbolic detectors for the NDA questions with the most errors (2026-10-04; neuro-symbolic: readable rules over the
compiled statements, fed to the stacker as features; each kept only if CV on the training NDAs improves).
Statements come from doccompile (lead-ins prefixed, page furniture out). Each detector returns a few named flags.
  nda-16 return / destroy WHEN THE AGREEMENT ENDS: a return/destroy duty and what triggers it (end of the agreement or
         of its purpose / a request or demand / nothing named)
  nda-20 may KEEP SOME information after returning it: a retention exception (archival, legal, backup copies) vs a
         plain "no copies retained"
Writes feats/nda_rules.jsonl ({id, flags...}) for nda_tune, nda_more, nda_ho."""
import json, re, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/doccompile"); sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/typed")

I = re.I
RET = re.compile(r"\b(?:return(?:ed|ing|s)?|destro(?:y|yed|ying|ys)|destruction|erase|erasure|delet(?:e|ed|ion)|expunge|purge|deliver(?:ed)? (?:up|back)|surrender)\b", I)
END = re.compile(r"\bterminat|\bexpir|\bend of\b|\bupon (?:the )?(?:completion|conclusion|cessation)|\bcompletion of\b|\bconclusion of\b|"
                 r"\bpurpose (?:is|has been) (?:achieved|fulfilled|completed|accomplished)|\bno longer (?:needed|required|necessary)|"
                 r"\b(?:once|when) the (?:purpose|project|evaluation|discussions|negotiations|transaction)\b|\bdecid\w+ not to (?:proceed|pursue)|"
                 r"\bdeadline\b|\bfail\w* to submit|\bnot awarded|\bwithin \w+ \(?\d*\)? ?(?:days|weeks|months) (?:after|following|of) (?:the )?(?:termination|expiration|end|completion)", I)
REQ = re.compile(r"\b(?:upon|on|at|following|after|within [^.;]{0,40}? (?:of|after|following)) (?:the )?(?:written )?(?:request|demand|direction|instruction|notice)\b|"
                 r"\b(?:if|when|whenever) (?:so )?(?:requested|demanded|asked|instructed)|\bat (?:any time|the) (?:upon )?(?:request|option|direction)|"
                 r"\brequest(?:ed)? (?:by|of|from) the\b", I)
KEEP_OK = re.compile(r"\b(?:may|is entitled to|shall be entitled to|is permitted to|shall be permitted to)\s+(?:\w+\s+){0,3}(?:retain|keep|maintain|store)|"
                     r"\b(?:except|other than|save|provided that)\b[^.;]{0,120}\b(?:retain|keep|archiv|one copy|single copy|back-?up|legal|compliance|regulatory|records)|"
                     r"\b(?:one|a single) (?:archival |file )?cop(?:y|ies)\b|\barchiv\w*|\bback-?up (?:tapes|copies|systems|files)|"
                     r"\bnot (?:be )?required to (?:return|destroy|delete|erase)\b", I)
KEEP_NO = re.compile(r"\b(?:shall|will|may) not (?:\w+ ){0,2}(?:retain|keep)\b|\bwithout (?:retaining|keeping)\b|\bretain no\b|\bno cop(?:y|ies)\b[^.;]{0,40}\b(?:retain|kept|remain)|"
                     r"\ball (?:copies|reproductions)\b[^.;]{0,80}\b(?:return|destroy)", I)


# nda-7 share with third parties (consultants, agents, advisors): a carve-out for them vs consent-only / a flat ban
THIRD = r"(?:third[- ]part(?:y|ies)|any (?:other )?(?:person|entity|party)|anyone|others?|outside (?:parties|agents)|agents?)"
ADVISORS = re.compile(r"\b(?:consultants?|advis[oe]rs?|attorneys?|counsel|accountants?|auditors?|agents?|representatives?|contractors?|"
                      r"lenders?|financing sources|affiliates?|professional)\b", I)
DISCLOSE = r"(?:disclos\w*|reveal\w*|divulg\w*|communicat\w*|make available|provid\w*|furnish\w*|shar\w*|transfer\w*|give)"
CONSENT_ONLY = re.compile(rf"\b(?:not|never|no)\b[^.;]{{0,120}}\b{DISCLOSE}\b[^.;]{{0,120}}\b{THIRD}\b[^.;]{{0,160}}\b(?:without|unless|except (?:with|after|upon))\b[^.;]{{0,60}}\b(?:prior )?(?:written )?(?:consent|approval|permission|agreement)|"
                          rf"\b{DISCLOSE}\b[^.;]{{0,80}}\b{THIRD}\b[^.;]{{0,80}}\b(?:must|shall) (?:first )?be (?:agreed|approved|consented)", I)
CARVE = re.compile(rf"\b(?:may|is permitted to|is entitled to|shall be permitted to|is free to|can)\b[^.;]{{0,80}}\b{DISCLOSE}\b|"
                   rf"\b(?:except|other than|save)\b[^.;]{{0,40}}\bto\b[^.;]{{0,120}}", I)
# nda-1 all CI expressly identified: definition requires marking / designation, vs covers what is confidential by nature or all information
MARKED = re.compile(r"\b(?:marked|labell?ed|designated|identified|stamped|legend|indicated|stated)\b[^.;]{0,40}\b(?:as )?(?:\W)?(?:confidential|proprietary|secret|restricted|sensitive)", I)
BY_NATURE = re.compile(r"\b(?:by (?:its|their) nature|reasonably (?:be )?(?:understood|considered|regarded|deemed)|ought reasonably|would reasonably|should (?:reasonably )?be treated|"
                       r"whether or not (?:so )?(?:marked|designated|identified|labell?ed)|regardless of whether|not marked|unmarked|whether (?:marked|designated) or not|"
                       r"(?:all|any) (?:and all )?information (?:disclosed|provided|furnished|received|obtained)\b(?![^.;]{0,60}\bmarked))", I)
# nda-2 technical only: the definition reaches commercial / business / financial information
COMMERCIAL = re.compile(r"\b(?:commercial|business|financial|marketing|sales|customers?|clients?|pricing|prices|costs?|suppliers?|personnel|employees?|"
                        r"strategic|strategies|plans?|forecasts?|accounts|profits?|revenues?|operations|legal|regulatory)\b", I)
TECHNICAL = re.compile(r"\b(?:technical|technology|know-how|software|source code|designs?|drawings|specifications|formulas?|processes|inventions?|"
                       r"research|data|algorithms?|engineering|prototypes?)\b", I)
DEFN = re.compile(r"\bconfidential information\b|\bproprietary information\b|\bevaluation material|\binformation\b", I)
DEF_VERB = re.compile(r"\b(?:means?|shall mean|includes?|shall include|refers? to|is defined as|consists? of)\b", I)
ALL_COPIES = re.compile(r"\b(?:return|destroy)\w*\b[^.;]{0,160}\b(?:all|any and all)\b[^.;]{0,80}\b(?:copies|reproductions|originals)\b|"
                        r"\b(?:all|any and all)\b[^.;]{0,40}\b(?:copies|reproductions)\b[^.;]{0,120}\b(?:return|destroy)", I)


def detect(text):
    from doccompile import compile_doc
    c = compile_doc(text)
    stmts = [re.sub(r"\s+", " ", s.text) for s in c.stmts if s.start >= 0]
    rets = [s for s in stmts if RET.search(s)]
    f16 = {"r16_has": bool(rets), "r16_end": any(END.search(s) for s in rets), "r16_req": any(REQ.search(s) for s in rets)}
    f16["r16_req_only"] = f16["r16_req"] and not f16["r16_end"]
    f16["r16_end_only"] = f16["r16_end"] and not f16["r16_req"]
    near = [s for s in stmts if RET.search(s) or re.search(r"\bretain|\bkeep\b|\barchiv|\bcop(?:y|ies)\b", s, I)]
    f20 = {"r20_keep_ok": any(KEEP_OK.search(s) for s in near), "r20_keep_no": any(KEEP_NO.search(s) for s in near), "r20_any": bool(near)}
    f20["r20_all_copies"] = any(ALL_COPIES.search(s) for s in rets)
    f20["r20_all_copies_no_keep"] = f20["r20_all_copies"] and not f20["r20_keep_ok"]
    third = [s for s in stmts if re.search(THIRD, s, I) and re.search(DISCLOSE, s, I)]
    f7 = {"r7_consent_only": any(CONSENT_ONLY.search(s) for s in third),
          "r7_advisor_carve": any(ADVISORS.search(s) and CARVE.search(s) for s in stmts if re.search(DISCLOSE, s, I)),
          "r7_any": bool(third)}
    f7["r7_consent_no_carve"] = f7["r7_consent_only"] and not f7["r7_advisor_carve"]
    defs = [s for s in stmts if DEFN.search(s) and DEF_VERB.search(s)][:6]
    f1 = {"r1_marked": any(MARKED.search(s) for s in defs), "r1_by_nature": any(BY_NATURE.search(s) for s in defs), "r1_defs": bool(defs)}
    f1["r1_marked_only"] = f1["r1_marked"] and not f1["r1_by_nature"]
    f2 = {"r2_commercial": any(COMMERCIAL.search(s) for s in defs), "r2_technical": any(TECHNICAL.search(s) for s in defs)}
    f2["r2_technical_only"] = f2["r2_technical"] and not f2["r2_commercial"]
    return {**f16, **f20, **f7, **f1, **f2}


def one(item):
    doc_id, text, qs = item
    try: f = detect(text)
    except Exception: f = {}
    return [{"id": f"{doc_id}/{q}", **f} for q in qs]


if __name__ == "__main__":
    import extract as X
    items = {}
    for s in ("nda_tune", "nda_more", "nda_ho"):
        for r in X.SETS[s]():
            d, q = r["id"].split("/")
            items.setdefault(d, [r["doc"], []])[1].append(q)
    with ProcessPoolExecutor(6) as ex, open("/root/zadumai_nli_proto/contract_map/typed/feats/nda_rules.jsonl", "w") as f:
        for rows in ex.map(one, [(d, t, qs) for d, (t, qs) in items.items()], chunksize=4):
            for r in rows: f.write(json.dumps(r) + "\n")
    print("done", len(items), "NDAs")

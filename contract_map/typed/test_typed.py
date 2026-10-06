"""Tests for the typed-pipeline work (2026-10-03): the question-typing changes in dev/router/frames.py and pipeline.py.
Run: ROUTER_PATH=.../typed/dev python -m pytest -q test_typed.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ROUTER_PATH", f"{HERE}/dev"))
sys.path.insert(0, HERE)

from router import frames  # noqa: E402
from router.pretier0 import check  # noqa: E402

import pipeline as P  # noqa: E402


def test_named_party_without_a_modal_after_it():
    doc = ("In the event of any stock split, the number of Shares shall be adjusted by the Committee, whose determination "
           "shall be conclusive. The Committee may adjust the number of Shares available under the Plan.")
    r = check("Can the Committee adjust the number of shares?", doc)
    assert not (r.frames or {}).get("reason", "").startswith("no frame")


def test_filler_word_between_actor_and_verb():
    t = frames._build_trie(frames._party_lexicon(["Borrower"]))
    assert not isinstance(frames.question_frame("Must the Borrower also deliver the security agreements?", t), str)


def test_negated_passive_subject_is_a_ban():
    doc = ("No benefit under the Plan shall in any manner or to any extent be assigned, alienated, or transferred by any "
           "Participant or Beneficiary.")
    r = check("Can a Participant assign the benefit under the Plan?", doc)
    assert r.answer != "yes"


def test_negative_subject_reader():
    raw = "no later than ten days the fee shall be paid".split()
    assert not frames._negative_subject(raw, 0, 8)
    raw = "neither this agreement nor any right shall be assigned".split()
    assert frames._negative_subject(raw, 1, 7)


def test_named_subject_beats_a_lexicon_thing_only_for_that_name():
    t = frames._build_trie(frames._party_lexicon(["Employee"]), "Employee")
    tags = frames.tag(frames.tokens("the employee may accrue vacation"), t)
    assert any(x.kind == "ACTOR" for x in tags)
    t2 = frames._build_trie(frames._party_lexicon(["Client"]))
    tags2 = frames.tag(frames.tokens("solicit any clients of the other party"), t2)
    assert not any(x.kind == "ACTOR" and x.name == "client" for x in tags2)


CONTRACT = "\n".join([
    "CONSULTING AGREEMENT",
    'This Consulting Agreement is made by and between SanDisk Corporation, a Delaware corporation ("SanDisk"), and '
    'John Smith, an individual ("Consultant").',
    "1. Services. Consultant shall perform the services described in Exhibit A. " + "The services are general. " * 30,
    "2. Fees. SanDisk shall pay Consultant the fees set out in Exhibit A within thirty days of invoice. " + "Fees are fixed. " * 30,
    "3. Subcontractors. Consultant shall not utilize third party contractors to perform the Services without SanDisk's "
    "prior written approval.",
    "4. Term. This Agreement shall continue for one year. " + "The term is fixed. " * 30,
    "5. Governing Law. This Agreement shall be governed by the laws of the State of California.",
])


def test_pipeline_locates_the_clause():
    pipe = P.Pipeline(net=False)
    cd = pipe.compiled(CONTRACT)
    assert any("third party contractors" in p["text"] for p in cd.pieces)
    assert 'the "Consultant"' in cd.party_line()
    frame = type("F", (), {"action": None, "actions": []})()
    idx = pipe.locate("Can the Consultant use subcontractors without approval?", CONTRACT, cd, frame)
    assert any("third party contractors" in cd.pieces[i]["text"] for i in idx)


def test_typed_check():
    fr = {"actions": ["NOTIFY"], "alts": []}
    assert P.typed_check(fr, "The Recipient shall notify the Discloser promptly.") is None
    assert P.typed_check(fr, "The obligations shall survive termination.") is not None
    assert P.typed_check({"actions": ["V:identifi"]}, "Information must be identified as confidential.") is None
    assert P.typed_check(None, "anything") is None


# ---------- the object model + query language (oo.py) ----------
import oo  # noqa: E402


def test_cql_parse_and_print():
    q = oo.parse('MAY(actor="Licensee", act=ASSIGN, object="this Agreement")')
    assert q.form == "MAY" and q.args == {"actor": "Licensee", "act": "ASSIGN", "object": "this Agreement"}
    assert str(q) == "MAY(actor='Licensee', act=ASSIGN, object='this Agreement')"
    assert oo.parse('HAS(clause="Governing Law")').args["clause"] == "Governing Law"
    import pytest
    with pytest.raises(ValueError):
        oo.parse("MAY(actor=Licensee)")  # MAY needs act
    with pytest.raises(ValueError):
        oo.parse("SELECT * FROM norms")


def test_question_to_query():
    fr = {"asks": "CAN", "actor": "RECEIVER", "actions": ["SHARE"], "things": [["THING", "CONFIDENTIAL_INFO"], ["THING", "EMPLOYEES"]]}
    q = oo.compile_question("May the receiving party share confidential information with its employees?", fr)
    assert (q.form, q.args["act"], q.args["object"]) == ("MAY", "SHARE", "EMPLOYEES")
    assert oo.compile_question("Does the agreement specify which law governs it?", {"asks": "EXISTS"}).args == {"clause": "Governing Law"}
    assert oo.compile_question("What happens if I'm late?", None).form == "ANSWER"


def test_contract_objects_and_clause_dispatch():
    c = oo.contract(CONTRACT)
    assert [p.role for p in c.parties] == ["", "consultant"] or any(p.name == "Consultant" for p in c.parties)
    assert any(s.heading == "Governing Law" for s in c.sections)
    assert isinstance(c.clause("Governing Law"), oo.GoverningLawClause)
    a = c.execute(oo.parse('HAS(clause="Governing Law")'), None)
    assert a is not None and a.value == "yes" and "California" in a.evidence
    v = c.execute(oo.parse('VALUE(clause="Governing Law", slot=law)'), None)
    assert v is not None and "California" in v.value

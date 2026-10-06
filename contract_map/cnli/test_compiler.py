"""Unit tests for the NDA compiler (compiler.py, ir.py, queries.py, queries_ir.py) (2026-10-03, overnight).
Run: cd cnli && ../../../../projects/zadumai/.venv/bin/python -m pytest -q test_compiler.py"""
import pytest

from compiler import compile_doc, parse, statements, _inline_split
from ir import facts, preprocess, _marking
import queries as Q
import queries_ir as QI


def test_lead_ins_are_prefixed_to_their_items():
    st = [s for s, _ in statements(parse("1. The Recipient shall not:\n(a) disclose the Confidential Information;\n(b) reverse engineer any sample."))]
    assert any(s.startswith("The Recipient shall not disclose") for s in st)
    assert any(s.startswith("The Recipient shall not reverse engineer") for s in st)


def test_inline_lists_split():
    lead, items = _inline_split("Confidential Information does not include information that: (a) is public; (b) is independently developed")
    assert lead.endswith("does not include information that") and items == ["is public", "is independently developed"]


def test_unmarked_items_follow_a_colon_and_stop_at_a_full_stop():
    text = ("Confidential Information does not include:\ninformation that is public;\ninformation independently developed by Recipient.\n"
            "The Recipient shall keep records.")
    p = compile_doc(text)
    assert any("independently developed" in e for e in p.ci_exclusions)
    assert not any("keep records" in e for e in p.ci_exclusions)


def test_definitions_quoted_and_bare():
    p = compile_doc('“Confidential Information” means all business and financial information of the Company.')
    assert p.ci_term == "Confidential Information" and p.ci_body
    p = compile_doc("Confidential Information means any technical data disclosed by Discloser.")
    assert p.ci_term and "technical data" in p.ci_body[0]


def test_preprocessor_canonical_verbs():
    assert "decompile" in preprocess("shall not reverse engineer the software")
    assert preprocess("Recipient agrees not to disclose").startswith("Recipient shall not disclose")
    assert "copy" in preprocess("Recipient may make copies of it")


def test_marking_necessary_vs_sufficient():
    assert _marking("information marked as confidential by the Discloser")[0] == "necessary"
    assert _marking("information marked as confidential or which a reasonable person would understand to be confidential")[0] == "sufficient"
    assert _marking("all information, whether or not marked as confidential")[0] == "sufficient"
    assert _marking("information designated as non-Confidential")[0] is None


def test_exception_to_a_ban_is_a_permission():
    f = facts("The Recipient shall not disclose the Confidential Information to any third party except to its employees and attorneys who need to know it.")
    assert QI.QUERIES["nda-5"](f)[0] == "yes" and QI.QUERIES["nda-7"](f)[0] == "yes"


def test_consent_gate_is_not_a_permission():
    f = facts("The Recipient shall not disclose the Confidential Information to any third party without the prior written consent of the Discloser.")
    r = QI.QUERIES["nda-7"](f)
    assert r is None or r[0] != "yes"


def test_return_on_request_only_is_not_return_upon_termination():
    f = facts("Upon written request of the Discloser, the Recipient shall return all Confidential Information.")
    assert QI.nda_16(f) is None
    f = facts("Upon termination of this Agreement, the Recipient shall promptly return or destroy all Confidential Information.")
    assert QI.nda_16(f)[0] == "yes"


def test_retention_proviso_is_permission_and_negated_retention_is_not():
    f = facts("Upon termination, the Recipient shall return or destroy all Confidential Information, provided that the Recipient may retain one archival copy for legal purposes.")
    assert QI.nda_20(f)[0] == "yes"
    f = facts("The Recipient will return all copies and will not retain any copies of the Confidential Information.")
    r = QI.nda_20(f)
    assert r is None or r[0] != "yes"


def test_no_license_and_not_construed_to_limit():
    assert QI.nda_15(facts("Nothing in this Agreement shall be construed as granting any license or right to the Confidential Information."))[0] == "no"
    assert QI.nda_15(facts("This Agreement shall not be construed to limit the Recipient's right to develop products independently.")) is None


def test_reverse_engineering_ban_from_a_list():
    f = facts("The Recipient shall not:\n(a) copy the samples;\n(b) reverse engineer, decompile or disassemble the samples.")
    assert QI.nda_11(f)[0] == "yes"


def test_survival_but_not_surviving_corporation():
    assert Q.nda_19(compile_doc("The obligations of this Agreement shall survive termination."))[0] == "yes"
    assert Q.nda_19(compile_doc("Principal shall be employed by the surviving corporation after the Merger.")) is None


def test_exclusions_answer_independent_development_and_third_parties_not_inclusions():
    p = compile_doc("The obligations shall not apply to information that (a) is independently developed by the Recipient; (b) is received from a third party without restriction.")
    assert Q.nda_12(p)[0] == "yes" and Q.nda_13(p)[0] == "yes"
    p = compile_doc("Confidential Information shall include information obtained by the Company from any third party.")
    assert Q.nda_13(p) is None

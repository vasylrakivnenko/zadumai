"""Unit tests for doccompile.py. Run: python -m pytest -q test_doccompile.py"""
from doccompile import compile_doc, preprocess, _split_inline

LICENSE = """Exhibit 10.4

SOFTWARE LICENSE AGREEMENT

This Software License Agreement (the "Agreement") is made as of May 1, 2020 by and between Alpha Software, Inc., a Delaware corporation ("Licensor"), and Beta Corp., a California corporation ("Licensee").

WHEREAS, Licensor owns certain software;

NOW, THEREFORE, the parties agree as follows:

ARTICLE 1 DEFINITIONS

1.1 "Software" means the program described in Exhibit A.

1.2 "Territory" means the United States.

ARTICLE 2 LICENSE

2.1 Grant. Licensor grants Licensee a non-exclusive license to use the Software in the Territory, subject to Section 2.2(a).

2.2 Restrictions. Licensee shall not:

(a) sublicense the Software; or

(b) reverse engineer the Software.

ARTICLE 3 GENERAL

3.1 Assignment. Neither party may assign this Agreement without the other's consent, except as set forth in Section 9.9.

3.2 Governing Law. This Agreement is governed by the laws of the State of New York. The Order Form is incorporated herein by reference.

EXHIBIT A

The Software: Alpha Analytics.
"""


def test_offsets_are_kept():
    text = "Intro text here.\n\nSource: ACME CORP, 10-K, 3/1/2019\n\n- 2 -\n\nMore text."
    clean, spans = preprocess(text)
    assert len(clean) == len(text)
    assert {w for *_, w in spans} == {"footer", "page"}
    assert "Source" not in clean and "- 2 -" not in clean
    assert clean.index("More text.") == text.index("More text.")


def test_inline_sections_follow_the_numbering():
    t = "1. Term. The term is one year. 2. Fees. Fees are due as stated in Section 4. The Company pays $5. 3. Notices. Notices go by mail."
    out = _split_inline(t)
    assert len(out) == len(t)
    lines = [l.strip() for l in out.split("\n")]
    assert lines[1].startswith("2. Fees") and lines[2].startswith("3. Notices")
    assert "Section 4. The Company" in out  # a reference is not a section start


def test_three_space_gaps_are_line_breaks():
    t = "the website.   3. Referral Fee   For each Approved Account, Affiliate earns a fee.   4. Approved Account   An account is approved when Chase approves it."
    c = compile_doc(t)
    heads = {s.number: s.heading for s in c.sections if s.number}
    assert heads.get("3") == "Referral Fee" and heads.get("4") == "Approved Account"


def test_glued_number():
    c = compile_doc("1. Scope. The scope is broad.\n5.Term of this Agreement\nThe term is one year and renews.")
    assert any(s.number == "5" for s in c.sections)


def test_structure_title_parties_and_headings():
    c = compile_doc(LICENSE)
    assert c.title == "SOFTWARE LICENSE AGREEMENT"
    assert [(p.name, p.role) for p in c.parties] == [("Licensor", "licensor"), ("Licensee", "licensee")]
    by = {s.number: s for s in c.sections if s.number}
    assert by["2.1"].heading == "Grant" and by["3.1"].heading == "Assignment"
    assert c.sections[by["2.1"].parent].heading.startswith("ARTICLE 2")
    st = next(s for s in c.stmts if "reverse engineer" in s.own)
    assert c.headings(st) == ["ARTICLE 2 LICENSE", "Restrictions"]
    assert "Licensee shall not reverse engineer" in st.text  # the lead-in is prefixed
    assert st.start >= 0 and LICENSE[st.start:st.end] == st.own
    assert not any(h in ("Exhibit 10.4", "SOFTWARE LICENSE AGREEMENT") for s in c.stmts for h in c.headings(s))


def test_symbols_and_xrefs():
    c = compile_doc(LICENSE)
    assert {"Software", "Territory", "Agreement"} <= set(c.symbols)
    x = {(r.kind, r.label): r for r in c.xrefs}
    assert x["section", "2.2(a)"].resolved and c.sections[x["section", "2.2(a)"].target].number == "2.2(a)"
    assert not x["section", "9.9"].resolved
    assert x["attachment", "Exhibit A"].resolved
    assert any(r.kind == "incorporation" for r in c.xrefs)


def test_table_of_contents_is_furniture():
    text = ("MASTER AGREEMENT\n\nTABLE OF CONTENTS\nARTICLE 1 DEFINITIONS 1\nARTICLE 2 SERVICES 3\nARTICLE 3 FEES 5\nARTICLE 4 TERM 7\n\n"
            "This Master Agreement is made by and between Acme, Inc. (\"Customer\") and Beta LLC (\"Supplier\") on the date below.\n\n"
            "ARTICLE 1 DEFINITIONS\n1.1 \"Services\" means the work.\nARTICLE 2 SERVICES\n2.1 Supplier shall perform the Services.")
    c = compile_doc(text)
    assert any(w == "toc" for *_, w in c.furniture)
    assert [(p.name, p.role) for p in c.parties] == [("Customer", "customer"), ("Supplier", "supplier")]
    arts = [s for s in c.sections if s.heading.startswith("ARTICLE")]
    assert [s.number for s in arts] == ["1", "2"]  # the contents' entries are not sections


def test_running_header_needs_four_pages_and_names_stay():
    page = "Body text of the page that goes on for a while and says something about the deal. " * 5
    hdr = "ACME CONFIDENTIAL DRAFT"
    text = "\n".join(f"{hdr}\n{page}" for _ in range(4)) + "\nACME HOLDINGS, INC.\n" + page + "\nACME HOLDINGS, INC.\n" + page + "\nACME HOLDINGS, INC.\n"
    clean, spans = preprocess(text)
    assert hdr not in clean and "ACME HOLDINGS, INC." in clean


def test_section_refs_into_other_documents():
    c = compile_doc("1. Scope. Subject to Section 7.09 of the Separation Agreement and Section 1 of this Agreement, see Section 2.e.\n"
                    "2. Fees. Fees are due monthly.")
    kinds = {(x.kind, x.label) for x in c.xrefs}
    assert ("document-section", "7.09 of the Separation Agreement") in kinds
    assert any(k == "section" and l == "1" for k, l in kinds)
    assert any(k == "section" and l == "2(e)" for k, l in kinds)

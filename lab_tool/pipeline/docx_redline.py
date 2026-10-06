"""Tracked-changes .docx writer for the pipeline (2026-10-06): word-level insertions / deletions in a paragraph, inserted
paragraphs, margin comments; optionally accept the document's existing tracked changes first (a counter-turn on the
other side's markup). Paragraph ids are contract_tool's ¶ ids: the same traversal as contract_tool.parse.parse_docx,
and every edit checks the paragraph's current text before touching it.
usage: from docx_redline import Redline; r = Redline(path, accept_existing=False); r.edit(i, new_text, comment=...);
       r.insert_after(i, text, comment=...); r.comment(i, text); r.save(out)"""
import copy, datetime, difflib, re, sys, zipfile
from lxml import etree
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from contract_tool.parse import W, _blocks, _clean, _runs, _views

NS_W = W[1:-1]
CT = "{http://schemas.openxmlformats.org/package/2006/content-types}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
COMMENTS_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
COMMENTS_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"


def _w(tag, **attrs):
    e = etree.Element(W + tag)
    for k, v in attrs.items(): e.set(W + k, str(v))
    return e


def _text_run(text, rpr=None, deleted=False):
    r = _w("r")
    if rpr is not None: r.append(copy.deepcopy(rpr))
    t = _w("delText" if deleted else "t"); t.text = text; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); r.append(t)
    return r


class Redline:
    def __init__(self, path, author="Reviewer", accept_existing=False):
        self.path, self.author = path, author
        self.date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        z = zipfile.ZipFile(path); self.files = {n: z.read(n) for n in z.namelist()}
        self.doc = etree.fromstring(self.files["word/document.xml"])
        self.units = self._map()  # unit index -> ("p", element, text) | ("row", element, text), as the parser numbers them
        self.next_id = 9000; self.comments = []; self.log = []
        if accept_existing: self.accept_all()

    def _map(self):
        out = []
        for b in _blocks(self.doc.find(W + "body")):
            if b.tag == W + "p":
                text, before, _, _ = _views(_runs(b))
                if text or before: out.append(["p", b, text])
            else:
                for tr in b.iter(W + "tr"):
                    views = [_views(_runs(tc)) for tc in tr.findall(W + "tc")]
                    if any(v[0] or v[1] for v in views): out.append(["row", tr, ""])
        return out

    def accept_all(self):
        """their tracked changes become plain text: deletions removed, insertions unwrapped, format changes dropped."""
        body = self.doc.find(W + "body")
        for tag in ("del", "moveFrom"):
            for e in list(body.iter(W + tag)):
                if e.getparent() is not None and e.getparent().tag != W + "rPr": e.getparent().remove(e)
        for tag in ("ins", "moveTo"):
            for e in list(body.iter(W + tag)):
                par = e.getparent()
                if par is None: continue
                if par.tag == W + "rPr": par.remove(e); continue
                i = par.index(e)
                for c in list(e): par.insert(i, c); i += 1
                par.remove(e)
        for tag in ("rPrChange", "pPrChange", "del"):  # paragraph-mark deletions and format changes
            for e in list(body.iter(W + tag)):
                if e.getparent() is not None: e.getparent().remove(e)
        for u in self.units:
            if u[0] == "p": u[2] = _views(_runs(u[1]))[0]

    def _id(self):
        self.next_id += 1; return self.next_id

    def para(self, i):
        if not (0 <= i < len(self.units)) or self.units[i][0] != "p": return None
        return self.units[i][1]

    def edit(self, i, new_text, comment=None):
        """rewrite paragraph ¶i as `new_text`, as word-level tracked changes. Returns True when applied."""
        p = self.para(i)
        if p is None: self.log.append(f"edit ¶{i}: not a paragraph"); return False
        old = self.units[i][2]; new = _clean(new_text)
        if not new or _clean(old) == new:
            if comment: self.comment(i, comment)
            return False
        runs = [r for r in p.iter(W + "r") if r.find(W + "t") is not None]
        rpr = runs[0].find(W + "rPr") if runs else None
        for c in list(p):  # keep paragraph properties, bookmarks and comment anchors; drop the text runs
            if c.tag in (W + "r", W + "hyperlink", W + "ins", W + "del", W + "smartTag", W + "fldSimple"): p.remove(c)
        A, B = old.split(), new.split(); first = True
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
            lead = "" if first else " "; first = False
            if tag == "equal":
                p.append(_text_run(lead + " ".join(A[i1:i2]), rpr)); continue
            if i2 > i1:
                d = _w("del", id=self._id(), author=self.author, date=self.date); d.append(_text_run(lead + " ".join(A[i1:i2]), rpr, deleted=True)); p.append(d); lead = " "
            if j2 > j1:
                ins = _w("ins", id=self._id(), author=self.author, date=self.date); ins.append(_text_run(lead + " ".join(B[j1:j2]), rpr)); p.append(ins)
        self.units[i][2] = new
        if comment: self.comment(i, comment)
        return True

    def insert_after(self, i, text, comment=None):
        """a new paragraph after ¶i (or at the end of the body when i is past the last unit), all of it inserted."""
        anchor = self.units[min(max(i, 0), len(self.units) - 1)][1] if self.units else None
        if anchor is not None and anchor.tag == W + "tr":  # after a table row: after the table
            while anchor.getparent() is not None and anchor.tag != W + "tbl": anchor = anchor.getparent()
        p = _w("p")
        ppr = anchor.find(W + "pPr") if anchor is not None and anchor.tag == W + "p" else None
        ppr = copy.deepcopy(ppr) if ppr is not None else _w("pPr")
        for c in ppr.findall(W + "rPr"): ppr.remove(c)
        rpr = _w("rPr"); rpr.append(_w("ins", id=self._id(), author=self.author, date=self.date)); ppr.append(rpr); p.append(ppr)
        ins = _w("ins", id=self._id(), author=self.author, date=self.date); ins.append(_text_run(_clean(text))); p.append(ins)
        if anchor is None: self.doc.find(W + "body").append(p)
        else: anchor.addnext(p)
        self.units.insert(i + 1, ["p", p, _clean(text)])  # later ids shift by one: callers insert from the end backwards
        if comment: self.comment(i + 1, comment)
        return True

    def comment(self, i, text):
        if not (0 <= i < len(self.units)): self.log.append(f"comment ¶{i}: out of range"); return False
        el = self.units[i][1]
        if el.tag == W + "tr":  # a table row: anchor on the first cell's first paragraph
            el = el.find(".//" + W + "p")
            if el is None: return False
        cid = len(self.comments); self.comments.append(_clean(text))
        el.insert(1 if el.find(W + "pPr") is not None else 0, _w("commentRangeStart", id=cid))
        el.append(_w("commentRangeEnd", id=cid))
        r = _w("r"); r.append(_w("commentReference", id=cid)); el.append(r)
        return True

    def _comments_xml(self):
        root = etree.Element(W + "comments", nsmap={"w": NS_W})
        for cid, text in enumerate(self.comments):
            c = _w("comment", id=cid, author=self.author, date=self.date, initials="R")
            for para in [t for t in text.split("\n") if t.strip()] or [""]:
                p = _w("p"); p.append(_text_run(para)); c.append(p)
            root.append(c)
        return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

    def save(self, out):
        files = dict(self.files)
        files["word/document.xml"] = etree.tostring(self.doc, xml_declaration=True, encoding="UTF-8", standalone=True)
        if self.comments:
            files["word/comments.xml"] = self._comments_xml()
            rels = etree.fromstring(files["word/_rels/document.xml.rels"])
            if not any(r.get("Type") == COMMENTS_TYPE for r in rels):
                n = 1 + max([int(m) for r in rels for m in re.findall(r"\d+", r.get("Id", ""))] or [0])
                e = etree.SubElement(rels, REL + "Relationship"); e.set("Id", f"rId{n}"); e.set("Type", COMMENTS_TYPE); e.set("Target", "comments.xml")
                files["word/_rels/document.xml.rels"] = etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True)
            ct = etree.fromstring(files["[Content_Types].xml"])
            if not any(o.get("PartName") == "/word/comments.xml" for o in ct.findall(CT + "Override")):
                o = etree.SubElement(ct, CT + "Override"); o.set("PartName", "/word/comments.xml"); o.set("ContentType", COMMENTS_CT)
                files["[Content_Types].xml"] = etree.tostring(ct, xml_declaration=True, encoding="UTF-8", standalone=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for n, data in files.items(): z.writestr(n, data)
        return out

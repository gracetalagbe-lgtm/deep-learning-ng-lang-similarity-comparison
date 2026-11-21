"""Build the publication-grade .docx from paper/paper.md.

Syntax handled (see paper.md):
- line 1 `# ...`            -> paper title
- [AUTHORS]/[AFFILIATION]    -> author block lines following each marker
- [ABSTRACT]/[KEYWORDS]      -> section + paragraph(s)
- `## `/`### `               -> headings
- [TCAP:...]                 -> table caption (italic, above the next pipe table)
- pipe tables                -> bordered tables (header bold)
- [FIG:file|caption]         -> centred image + caption (from paper/figures/)
- [EQ]...[/EQ]               -> centred italic display line
- [CASESTUDY_TABLE]          -> Table XI rendered verbatim from paper/case_study.json
- `## References`            -> hanging-indent reference entries
- inline **bold** / *italic* ; lists ("- ", "1. ")

Output: paper/paper.docx (then converted to PDF by LibreOffice externally).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MD = (HERE / "paper.md").read_text(encoding="utf-8")
CASE = json.loads((HERE / "case_study.json").read_text(encoding="utf-8"))

BODY_FONT = "Cambria"
HEAD_FONT = "Calibri"


def add_runs(par, text: str, size: float | None = None) -> None:
    """Render **bold** / *italic* inline markup into runs."""
    tokens = re.split(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            run = par.add_run(tok[2:-2])
            run.bold = True
        elif tok.startswith("`") and tok.endswith("`"):
            run = par.add_run(tok[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt((size or 10.5) - 1)
            continue
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            run = par.add_run(tok[1:-1])
            run.italic = True
        else:
            run = par.add_run(tok)
        if size:
            run.font.size = Pt(size)


def set_cell(cell, text: str, bold: bool = False, size: float = 8.5) -> None:
    cell.text = ""
    par = cell.paragraphs[0]
    par.paragraph_format.space_before = Pt(1)
    par.paragraph_format.space_after = Pt(1)
    add_runs(par, text, size=size)
    for run in par.runs:
        run.font.size = Pt(size)
        if bold:
            run.bold = True


def style_table(table, size: float = 8.5) -> None:
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in table.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.space_before = Pt(1)
                par.paragraph_format.space_after = Pt(1)
                for run in par.runs:
                    run.font.size = Pt(size)
                    run.font.name = BODY_FONT


def caption_par(doc, text: str) -> None:
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    par.paragraph_format.space_before = Pt(8)
    par.paragraph_format.space_after = Pt(3)
    add_runs(par, text, size=9)
    for run in par.runs:
        run.italic = True


def fig_par(doc, filename: str, cap: str) -> None:
    path = HERE / "figures" / filename
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(8)
    run = par.add_run()
    run.add_picture(str(path), width=Inches(6.2))
    cpar = doc.add_paragraph()
    cpar.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cpar.paragraph_format.space_after = Pt(10)
    add_runs(cpar, cap, size=9)
    for run in cpar.runs:
        run.italic = True


def case_study_table(doc) -> None:
    cols = ["Lang", "Case", "Sentence A (…)", "Sentence B (…)", "Gold", "Raw", "Cal", "Verdict"]
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(cols):
        set_cell(table.rows[0].cells[i], c, bold=True)
    kinds = {"true_positive": ("TP", "paraphrase"), "false_positive": ("FP", "false positive"),
             "false_negative": ("FN", "false negative"), "true_negative": ("TN", "not paraphrase")}
    verdicts = {"true_positive": "correct", "false_positive": "error (over-prediction)",
                "false_negative": "error (missed)", "true_negative": "correct"}
    for lang in ["hau", "yor", "ibo"]:
        cfg = CASE[lang]
        tau = f"τ={cfg['threshold']:.2f}"
        for cse in cfg["cases"]:
            row = table.add_row()
            klabel, _ = kinds[cse["kind"]]
            gold = str(cse["gold"])
            if cse.get("gold_score_continuous") is not None:
                gold += f" ({cse['gold_score_continuous']:.2f})"
            a = cse["sentence1"].replace("\n", " ")
            b = cse["sentence2"].replace("\n", " ")
            set_cell(row.cells[0], lang)
            set_cell(row.cells[1], klabel)
            set_cell(row.cells[2], (a[:44] + "…") if len(a) > 45 else a)
            set_cell(row.cells[3], (b[:44] + "…") if len(b) > 45 else b)
            set_cell(row.cells[4], gold)
            set_cell(row.cells[5], f"{cse['raw_prob']:.3f}")
            set_cell(row.cells[6], f"{cse['calibrated_prob']:.3f}")
            set_cell(row.cells[7], f"{verdicts[cse['kind']]} ({tau})")
    style_table(table, size=7.5)


def add_page_number(par) -> None:
    run = par.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    par._p.append(fld)


def main() -> None:
    doc = Document()
    # page + base style
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(1)
    sec.top_margin = sec.bottom_margin = Inches(1)
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    # header / footer
    hp = sec.header.paragraphs[0]
    hp.text = "Low-Resource Paraphrase Detection for Yorùbá, Igbo and Hausa"
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in hp.runs:
        run.font.size = Pt(8)
        run.font.name = HEAD_FONT
        run.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(fp)
    for run in fp.runs:
        run.font.size = Pt(9)

    lines = MD.split("\n")
    i = 0
    first_h1_done = False
    table_cap_pending = None
    in_refs = False

    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        if not line.strip():
            continue

        if line.startswith("[AUTHORS]"):
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(par, lines[i].strip(), size=12)
            i += 1
            continue
        if line.startswith("[AFFILIATION]"):
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(par, lines[i].strip(), size=10)
            i += 1
            continue
        if line == "[ABSTRACT]":
            h = doc.add_heading("Abstract", level=1)
            for run in h.runs:
                run.font.name = HEAD_FONT
            continue
        if line == "[KEYWORDS]":
            h = doc.add_heading("Keywords", level=1)
            for run in h.runs:
                run.font.name = HEAD_FONT
            continue
        if line == "[CASESTUDY_TABLE]":
            case_study_table(doc)
            continue
        if line.startswith("[TCAP:"):
            caption_par(doc, line[6:-1])
            continue
        if line.startswith("[EQ]"):
            eq = line[4:-5].strip() if line.endswith("[/EQ]") else line[4:].strip()
            while not lines[i - 1].endswith("[/EQ]"):
                eq += " " + lines[i].strip()
                i += 1
            eq = eq.replace("[/EQ]", "").strip()
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_before = Pt(4)
            par.paragraph_format.space_after = Pt(8)
            run = par.add_run(eq)
            run.italic = True
            run.font.size = Pt(10.5)
            continue
        if line.startswith("[FIG:"):
            body = line[5:-1]
            fname, cap = body.split("|", 1)
            fig_par(doc, fname.strip(), cap.strip())
            continue

        if line.startswith("# ") and not first_h1_done:
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_after = Pt(12)
            run = par.add_run(line[2:])
            run.bold = True
            run.font.size = Pt(16)
            run.font.name = HEAD_FONT
            first_h1_done = True
            continue

        if line.startswith("## "):
            title = line[3:]
            if title == "References":
                in_refs = True
                doc.add_page_break()
            elif title.startswith("1. Introduction"):
                doc.add_page_break()
            h = doc.add_heading(title, level=1)
            for run in h.runs:
                run.font.name = HEAD_FONT
                run.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B)
            continue
        if line.startswith("### "):
            h = doc.add_heading(line[4:], level=2)
            for run in h.runs:
                run.font.name = HEAD_FONT
                run.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B)
            continue

        if line.startswith("|"):
            rows = [[c.strip() for c in line.strip().strip("|").split("|")]]
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            rows = [r for r in rows if not all(set(c) <= {"-", ":", " "} for c in r)]
            ncols = max(len(r) for r in rows)
            table = doc.add_table(rows=0, cols=ncols)
            table.autofit = True
            for ri, r in enumerate(rows):
                cells = table.add_row().cells
                for ci in range(ncols):
                    set_cell(cells[ci], r[ci] if ci < len(r) else "", bold=(ri == 0))
            style_table(table)
            # proportional column widths (content-driven), fixed layout
            totals = []
            for j in range(ncols):
                m = max((len(row.cells[j].text) for row in table.rows), default=4)
                totals.append(max(m, 4))
            tot = sum(totals)
            table.autofit = False
            tblPr = table._tbl.tblPr
            layout = OxmlElement("w:tblLayout")
            layout.set(qn("w:type"), "fixed")
            tblPr.append(layout)
            base = 0.5
            excess = [max(t - 4, 0) for t in totals]
            denom = sum(excess) or 1
            remaining = max(6.4 - base * ncols, 0.1)
            widths = [base + remaining * e / denom for e in excess]
            grid = table._tbl.find(qn("w:tblGrid"))
            for gc, j in zip(grid.findall(qn("w:gridCol")), range(ncols)):
                gc.set(qn("w:w"), str(int(widths[j] * 1440)))
            for j in range(ncols):
                for row in table.rows:
                    row.cells[j].width = Inches(widths[j])
            sp = doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(4)
            continue

        if re.match(r"^(\d+\.|-)\s", line):
            style = "List Number" if line[0].isdigit() else "List Bullet"
            par = doc.add_paragraph(style=style)
            add_runs(par, re.sub(r"^(\d+\.|-)\s+", "", line))
            continue

        if in_refs and line.strip() and not line.startswith("#"):
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Inches(0.35)
            par.paragraph_format.first_line_indent = Inches(-0.35)
            par.paragraph_format.space_after = Pt(4)
            add_runs(par, line, size=9.5)
            continue

        par = doc.add_paragraph()
        add_runs(par, line)

    # page break before Section 1 (after keywords block)
    out = HERE / "paper.docx"
    doc.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

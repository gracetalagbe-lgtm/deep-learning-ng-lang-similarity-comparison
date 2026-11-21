"""Build the 10-slide academic presentation (16:9, navy theme) with python-pptx.

Every quantitative claim is grounded in the project's artifacts (see
presentation_research_notes.md). Charts are pre-rendered 300-DPI transparent
PNGs (paper/make_slide_charts.py); the architecture diagram is drawn with
native editable shapes.

Output: paper/deep_learning_project_presentation.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "presentation_assets"

NAVY = RGBColor(0x0D, 0x1B, 0x2A)
PANEL = RGBColor(0x12, 0x24, 0x3B)
PANEL2 = RGBColor(0x16, 0x2E, 0x4A)
INK = RGBColor(0xEA, 0xEF, 0xF6)
INK2 = RGBColor(0xA8, 0xB4, 0xC4)
ACCENT = RGBColor(0x39, 0x87, 0xE5)
CYAN = RGBColor(0x7C, 0xC4, 0xF7)
GOLD = RGBColor(0xC9, 0x85, 0x00)
FONT = "Calibri"

W, H = Inches(13.333), Inches(7.5)
TITLE_W = W - Inches(1.0)

prs = Presentation()
prs.slide_width = W
prs.slide_height = H
BLANK = prs.slide_layouts[6]


def add_slide() -> object:
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY
    return s


def tb(slide, x, y, w, h, text="", size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
       italic=False, spacing=1.0, font=FONT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    r = p.add_run()
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.name = font
    f.color.rgb = color
    return box


def bullets(slide, x, y, w, h, items, size=12.5, gap=6, color=INK, marker="•  ", mcolor=CYAN,
            title=None, title_size=13):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    if title:
        p = tf.paragraphs[0]
        p.space_after = Pt(gap + 2)
        r = p.add_run()
        r.text = title
        r.font.size = Pt(title_size)
        r.font.bold = True
        r.font.color.rgb = CYAN
        r.font.name = FONT
        first = False
    for it in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = 1.04
        rm = p.add_run()
        rm.text = marker
        rm.font.size = Pt(size)
        rm.font.color.rgb = mcolor
        rm.font.name = FONT
        if isinstance(it, tuple):  # (bold lead-in, rest)
            rb = p.add_run()
            rb.text = it[0]
            rb.font.size = Pt(size)
            rb.font.bold = True
            rb.font.color.rgb = color
            rb.font.name = FONT
            rr = p.add_run()
            rr.text = it[1]
            rr.font.size = Pt(size)
            rr.font.color.rgb = color
            rr.font.name = FONT
        else:
            rr = p.add_run()
            rr.text = it
            rr.font.size = Pt(size)
            rr.font.color.rgb = color
            rr.font.name = FONT
    return box


def panel(slide, x, y, w, h, fill=PANEL, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.adjustments[0] = 0.045
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.0)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def header(slide, title, kicker=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.42), Inches(0.14), Inches(0.46))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    tb(slide, Inches(0.78), Inches(0.30), TITLE_W, Inches(0.7), title, size=26, bold=True)
    if kicker:
        tb(slide, Inches(0.80), Inches(0.92), TITLE_W, Inches(0.3), kicker, size=11.5, color=INK2)


def footer(slide, n):
    tb(slide, Inches(0.5), H - Inches(0.42), Inches(6), Inches(0.3),
       "Transformer Semantic Similarity & Paraphrase Detection — Yoruba · Igbo · Hausa",
       size=8.5, color=INK2)
    tb(slide, W - Inches(1.2), H - Inches(0.42), Inches(0.7), Inches(0.3), f"{n} / 10",
       size=9, color=INK2, align=PP_ALIGN.RIGHT)


def arrow(slide, x, y, w=Inches(0.34), h=Inches(0.26)):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    a.fill.solid()
    a.fill.fore_color.rgb = ACCENT
    a.line.fill.background()
    a.shadow.inherit = False
    return a


def shape_box(slide, x, y, w, h, title, body=None, fill=PANEL2, line=ACCENT,
              tsize=12, bsize=10.5, tcolor=INK, bcolor=INK2):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(1.25)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = title
    r.font.size = Pt(tsize)
    r.font.bold = True
    r.font.color.rgb = tcolor
    r.font.name = FONT
    if body:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(3)
        p2.line_spacing = 1.02
        r2 = p2.add_run()
        r2.text = body
        r2.font.size = Pt(bsize)
        r2.font.color.rgb = bcolor
        r2.font.name = FONT
    return sh


def table(slide, x, y, w, rows_data, col_widths=None, size=10.5, header_size=10.5):
    nrows, ncols = len(rows_data), len(rows_data[0])
    gt = slide.shapes.add_table(nrows, ncols, x, y, w, Inches(0.32 * nrows))
    t = gt.table
    if col_widths:
        total = sum(col_widths)
        for j, cw in enumerate(col_widths):
            t.columns[j].width = Emu(int(w * cw / total))
    for i, row in enumerate(rows_data):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL2 if i == 0 else PANEL
            p = cell.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(header_size if i == 0 else size)
            r.font.bold = (i == 0)
            r.font.color.rgb = CYAN if i == 0 else INK
            r.font.name = FONT
    return t


# ---------------------------------------------------------------- Slide 1
s = add_slide()
# subtle transformer/attention motif: two token rows with attention arcs (thin lines + dots)
motif_y = Inches(6.1)
toks = [Inches(9.4) + i * Inches(0.72) for i in range(5)]
for i, tx in enumerate(toks):
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, tx, motif_y, Inches(0.16), Inches(0.16))
    c.fill.solid()
    c.fill.fore_color.rgb = ACCENT if i % 2 == 0 else CYAN
    c.line.fill.background()
    c.shadow.inherit = False
for i in range(len(toks) - 1):
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, toks[i] + Inches(0.14), motif_y + Inches(0.065),
                            Inches(0.60), Pt(1.1))
    ln.fill.solid()
    ln.fill.fore_color.rgb = PANEL2
    ln.line.fill.background()
    ln.shadow.inherit = False
ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.52), Inches(11.33), Pt(1.4))
ln.fill.solid()
ln.fill.fore_color.rgb = ACCENT
ln.line.fill.background()
ln.shadow.inherit = False

tb(s, Inches(1.0), Inches(0.85), Inches(11.3), Inches(0.4), "DEEP LEARNING PROJECT",
   size=13, color=CYAN, bold=True)
tb(s, Inches(1.0), Inches(1.85), Inches(11.33), Inches(2.1),
   "A Transformer-Based Approach for Semantic Similarity and\nParaphrase Detection in Yoruba, Igbo, and Hausa Texts",
   size=30, bold=True, spacing=1.12)
tb(s, Inches(1.0), Inches(4.05), Inches(11.3), Inches(0.4),
   "Fine-tuned AfroXLMR cross-encoders · leakage-controlled evaluation · temperature-calibrated serving",
   size=13.5, color=INK2, italic=True)
tb(s, Inches(1.0), Inches(4.75), Inches(11.3), Inches(1.2),
   "Grace Toluwani Alagbe  —  Matriculation Number: 262092", size=15, bold=True)
tb(s, Inches(1.0), Inches(5.18), Inches(11.3), Inches(0.4),
   "Adigun Oluwaseunayo Elijah  —  Matriculation Number: 262020", size=15, bold=True)
tb(s, Inches(1.0), Inches(5.78), Inches(11.3), Inches(0.4),
   "Submitted to:  Dr Adeleke", size=14, color=INK2)
footer(s, 1)

# ------------------------------------------- Slide 2: background/problem/objectives
s = add_slide()
header(s, "Background, Problem Statement and Objectives")
colw = Inches(3.94)
x0, x1, x2 = Inches(0.5), Inches(4.70), Inches(8.90)
panel(s, x0, Inches(1.42), colw, Inches(5.42))
panel(s, x1, Inches(1.42), colw, Inches(5.42))
panel(s, x2, Inches(1.42), colw, Inches(5.42))
bullets(s, x0 + Inches(0.22), Inches(1.62), colw - Inches(0.44), Inches(5.1),
        [("Semantic similarity ", "judges meaning equivalence, not word overlap."),
         ("Paraphrase detection ", "supports search, plagiarism and FAQ matching."),
         ("Yoruba, Igbo, Hausa ", "are spoken by 100M+ people but low-resource in NLP."),
         ("No paraphrase dataset existed ", "for these languages (verified vs HF, GitHub, NaijaNLP 2025)."),
         ("Transformers ", "give contextual representations that survive rewording.")],
        title="Background", size=11.5, gap=7)
bullets(s, x1 + Inches(0.22), Inches(1.62), colw - Inches(0.44), Inches(5.1),
        [("No labelled pair data: ", "labels had to be constructed (MT-inherited “silver”)."),
         ("Split leakage: ", "official Hausa split shares 644 train∩test sentences (62%)."),
         ("Word-overlap baselines ", "collapse on high-overlap, different-meaning pairs."),
         ("Orthography: ", "Yoruba tone marks are phonemic; marking is often mixed."),
         ("Silver-label risk: ", "translation can erase the distinction a label encodes.")],
        title="Problem Statement", size=11.5, gap=7)
bullets(s, x2 + Inches(0.22), Inches(1.62), colw - Inches(0.44), Inches(5.1),
        [("1.  ", "Build provenance-audited pair datasets and a sentence-disjoint split."),
         ("2.  ", "Fine-tune and evaluate transformer cross-encoders in three languages."),
         ("3.  ", "Quantify effects of split design and label provenance."),
         ("4.  ", "Deploy a calibrated, provenance-transparent demo application.")],
        title="Research Objectives", size=11.5, gap=9, marker="")
footer(s, 2)

# ------------------------------------------------- Slide 3: architecture
s = add_slide()
header(s, "System Architecture and Methodology",
       "Implemented pipeline — training on Colab T4; scoring, calibration and serving on CPU")
lay_y = Inches(1.75)
labels = ["INPUT", "PROCESSING", "MODEL", "PREDICTION", "OUTPUT"]
lx = [Inches(0.55), Inches(2.85), Inches(5.15), Inches(7.95), Inches(10.75)]
lw = [Inches(1.9), Inches(1.9), Inches(2.4), Inches(2.4), Inches(2.05)]
for lab, x, w in zip(labels, lx, lw):
    tb(s, x, lay_y - Inches(0.34), w, Inches(0.28), lab, size=10.5, color=CYAN, bold=True)
shape_box(s, lx[0], Inches(2.35), lw[0], Inches(1.15),
          "Sentence pair", "Yoruba · Igbo · Hausa\n(real held-out examples)")
arrow(s, Inches(2.47), Inches(2.80))
shape_box(s, lx[1], Inches(2.10), lw[1], Inches(1.15),
          "clean_text", "NFC · strip U+200B/200E/\n200F/FEFF · whitespace")
shape_box(s, lx[1], Inches(3.55), lw[1], Inches(1.15),
          "Language routing", "3 task tracks\n(+ English transfer arm)")
shape_box(s, lx[1], Inches(5.00), lw[1], Inches(1.15),
          "Tokenise", "SentencePiece · maxlen 128")
arrow(s, Inches(4.77), Inches(2.80))
arrow(s, Inches(4.77), Inches(4.25))
arrow(s, Inches(4.77), Inches(5.55))
shape_box(s, lx[2], Inches(2.55), lw[2], Inches(1.5),
          "AfroXLMR-base cross-encoder", "fine-tuned · 278,045,186 params\nfp16 · Colab T4 · 3 seeds", fill=RGBColor(0x1A, 0x3B, 0x63))
shape_box(s, lx[2], Inches(4.55), lw[2], Inches(1.25),
          "Bi-encoder track", "LaBSE / AfriE5\nunit-norm cosine similarity")
arrow(s, Inches(7.57), Inches(3.30))
arrow(s, Inches(7.57), Inches(5.15))
shape_box(s, lx[3], Inches(2.35), lw[3], Inches(1.1),
          "softmax → log-odds z", "positive-class probability")
shape_box(s, lx[3], Inches(3.75), lw[3], Inches(1.1),
          "Temperature scaling", "calibrated p̂ = σ(z / T), T fit on dev")
shape_box(s, lx[3], Inches(5.15), lw[3], Inches(1.1),
          "Dev-tuned threshold τ", "99-point grid · never tuned on test")
arrow(s, Inches(10.37), Inches(2.90))
arrow(s, Inches(10.37), Inches(5.65))
shape_box(s, lx[4], Inches(2.55), lw[4], Inches(1.35),
          "Verdict + calibrated score", "paraphrase / not-paraphrase")
shape_box(s, lx[4], Inches(4.45), lw[4], Inches(1.5),
          "Streamlit demo", "provenance panel per model\n(3 languages served)")
footer(s, 3)

# ------------------------------------------- Slide 4: dataset & preparation
s = add_slide()
header(s, "Datasets and Data Preparation",
       "All counts from the processed CSVs; label provenance tracked per pair")
rows = [
    ["Dataset", "Pairs", "Train / Dev / Test", "Positive %", "Label provenance"],
    ["SemRel2024-Hausa (official)", "2,551", "1,736 / 212 / 603", "56.3", "human (BWS 0–1, ≥0.5)"],
    ["SemRel-Hausa disjoint re-split", "2,551", "2,041 / 255 / 255", "54.1 / 65.1 / 65.1", "human"],
    ["Yoruba silver (translate-train)", "2,873", "2,299 / 287 / 287", "45.1", "silver — PAWS via NLLB MT"],
    ["Igbo silver (translate-train)", "2,948", "2,358 / 295 / 295", "44.7", "silver — PAWS via NLLB MT"],
    ["Hausa silver (translate-train)", "1,477", "1,181 / 148 / 148", "43.9", "silver — PAWS via NLLB MT"],
    ["AfriXNLI binary eval (yo/ig/ha/en)", "700/697/700/700", "300 / 300 / 300", "50.0", "proxy (entail→1, contra→0)"],
]
table(s, Inches(0.5), Inches(1.5), Inches(12.33), rows, col_widths=[3.1, 1.5, 1.9, 1.6, 3.2], size=10.5)
s.shapes.add_picture(str(ASSETS / "s4_datasets.png"), Inches(0.55), Inches(4.05), width=Inches(9.1))
bullets(s, Inches(9.95), Inches(4.15), Inches(2.95), Inches(2.9),
        ["Preprocessing: NFC, zero-width char stripping, diacritics preserved",
         "Tokenisation: pair-encoded, truncation measured (44.8% of Yoruba pairs > 128 tk)",
         "Silver labels: copy-through guard, dedup, length-ratio QC",
         "Igbo eval = 697: 10 malformed dev TSV lines (1.1% loss)"],
        size=10, gap=5)
footer(s, 4)

# --------------------------------------------- Slide 5: evaluation framework
s = add_slide()
header(s, "Evaluation Framework",
       "Metrics chosen for binary paraphrase classification + graded relatedness scoring")
boxes = [
    ("1 · Classification performance", Inches(0.5), Inches(1.55),
     ["Accuracy, Precision, Recall, F1 (dev-tuned threshold)",
      "ROC-AUC (threshold-free)",
      "Confusion matrices at deployed thresholds"]),
    ("2 · Ranking / similarity", Inches(4.83), Inches(1.55),
     ["Spearman ρ — primary (BWS gold is ordinal-leaning)",
      "Pearson r as secondary",
      "Undefined on silver sets (binary-only labels)"]),
    ("3 · Significance & robustness", Inches(9.16), Inches(1.55),
     ["McNemar exact test vs TF-IDF baseline",
      "Paired bootstrap CIs (10,000 / 2,000 resamples)",
      "3 seeds (42/43/44) mean ± std; official vs disjoint splits"]),
    ("4 · Calibration & efficiency", Inches(0.5), Inches(4.30),
     ["Temperature scaling on dev (Guo et al., 2017); 10-bin ECE",
      "Training: 263–1,147 s per run on T4, fp16 (~5.7 h for 29 runs)",
      "Checkpoint 1.06 GB; CPU inference < 1 s (load ≈ 10 s)"]),
    ("5 · Language-level & generalisation", Inches(4.83), Inches(4.30),
     ["Per-language test folds (yo / ig / ha)",
      "Diacritics-stripping ablation (paired, threshold held)",
      "Lexical-overlap buckets; cross-lingual EN→target transfer"]),
]
for title, x, y, items in boxes:
    h = Inches(2.55) if y < Inches(3) else Inches(2.35)
    panel(s, x, y, Inches(3.67), h)
    bullets(s, x + Inches(0.2), y + Inches(0.16), Inches(3.3), h - Inches(0.3), items,
            title=title, size=10.5, gap=5)
tb(s, Inches(0.5), Inches(6.72), Inches(12.3), Inches(0.4),
   "Why these metrics: F1 balances the false-positive skew of silver data; Spearman matches the "
   "ordinal Best-Worst-Scaling gold; McNemar + bootstrap guard against small-test-set overclaiming.",
   size=10.5, color=INK2, italic=True)
footer(s, 5)

# ------------------------------------------------------ Slide 6: results
s = add_slide()
header(s, "Results — Experimental Performance",
       "Human-labelled Hausa (SemRel2024); leakage-free disjoint test, n = 255")
s.shapes.add_picture(str(ASSETS / "s6_ranking.png"), Inches(0.5), Inches(1.75), width=Inches(6.7))
s.shapes.add_picture(str(ASSETS / "s6_confusion.png"), Inches(7.55), Inches(1.85), width=Inches(3.55))
panel(s, Inches(11.35), Inches(1.75), Inches(1.48), Inches(3.1), fill=PANEL2)
tb(s, Inches(11.42), Inches(1.9), Inches(1.36), Inches(2.9),
   "0.783\naccuracy\n\n0.863\nROC-AUC\n\n0.697\nSpearman ρ\n(3 seeds)",
   size=13.5, bold=True, color=CYAN, align=PP_ALIGN.CENTER, spacing=1.05)
bullets(s, Inches(0.55), Inches(5.15), Inches(12.2), Inches(1.8),
        [("Headline: ", "fine-tuned AfroXLMR cross-encoder beats the lexical baseline by +13.2 points (McNemar exact p = 8.1×10⁻⁶) and matches the SemRel2024 paper's fine-tuned LaBSE (ρ 0.69)."),
         ("Leakage: ", "official-split scores are inflated ≈ +6 points and invert model selection (LaBSE-FT wins the leaky split, loses the clean split)."),
         ("Transfer: ", "English-PAWS zero-shot (0.811) beats in-language silver fine-tuning (0.626) on Hausa — clean labels travel better.")],
        size=11, gap=6)
footer(s, 6)

# ------------------------------------------- Slide 7: model comparison
s = add_slide()
header(s, "Model Comparison and Analysis",
       "All evaluated systems; selection justified by verified coverage and measured performance")
rows = [
    ["Model", "Params", "Pretraining / coverage", "Role in project", "Observed result"],
    ["AfroXLMR-base", "278,045,186 (counted)", "XLM-R-base + adaptive FT on 17 African languages (MIT)", "primary cross-encoder", "best clean-split classifier (0.783 acc)"],
    ["AfriBERTa-large", "≈126M", "African news + CommonCrawl <1 GB (MIT)", "African-specific baseline", "fails on silver data (predict-all)"],
    ["LaBSE", "≈471M", "109-language bitext, yo/ig/ha verified (Apache-2.0)", "embedding track + fine-tune arm", "ρ 0.604 FT / 0.476 zero-shot"],
    ["AfriE5-large-instruct", "≈560M", "African-specialised; query:/passage: prefixes (MIT)", "embedding track (zero-shot)", "ρ 0.590 — best zero-shot"],
    ["mBERT", "≈178M", "104 languages — Igbo & Hausa absent", "excluded (negative control only)", "not benchmarked"],
]
table(s, Inches(0.5), Inches(1.5), Inches(8.2), rows, col_widths=[1.5, 1.5, 2.6, 1.9, 2.2], size=9.5, header_size=10)
s.shapes.add_picture(str(ASSETS / "s7_radar.png"), Inches(8.95), Inches(1.55), height=Inches(4.4))
bullets(s, Inches(0.55), Inches(4.62), Inches(8.1), Inches(2.2),
        [("Why AfroXLMR: ", "multilingual adaptive fine-tuning covers all three targets with verified licence, and it is the architecture family of the SemEval-2024 Task 1 winning system."),
         ("Radar uses only measured metrics ", "(accuracy / F1 / ROC-AUC / Spearman, disjoint Hausa test) — no subjective scores; TF-IDF included as the lexical reference."),
         ("mBERT excluded: ", "fine-tuning it on Igbo/Hausa would measure tokenisation OOV behaviour, not semantics.")],
        size=10.5, gap=5)
footer(s, 7)

# --------------------------------------- Slide 8: language-level analysis
s = add_slide()
header(s, "Language-Level Analysis and Discussion",
       "Silver-track test folds; examples are genuine held-out pairs from the deployed predictor")
s.shapes.add_picture(str(ASSETS / "s8_language.png"), Inches(0.5), Inches(1.6), width=Inches(6.5))
ex_rows = [
    ["Lang", "Sentence A → Sentence B (abridged)", "Gold", "Prediction"],
    ["hau", "“Gina Jami'oi masu zaman kansu…” → “Sarkin Bayerce gina irin wadannan…”", "1", "paraphrase ✓ (0.927)"],
    ["yor", "“Àwọn Kátólíìkì tó kọ́ láti di Kristẹni…” → “Ìwé ìròyìn sọ pé wọ́n jẹ́ ọmọ ilẹ̀…”", "0", "paraphrase ✗ (0.703)"],
    ["hau", "“A lokacin jawabin sa, Uba Sani…” → “Uba Sani ya ba da shawarar…”", "1", "not paraphrase ✗ (0.124)"],
]
table(s, Inches(7.3), Inches(1.7), Inches(5.5), ex_rows, col_widths=[0.7, 3.4, 0.6, 1.3], size=9, header_size=9.5)
bullets(s, Inches(7.3), Inches(3.35), Inches(5.55), Inches(3.4),
        [("Igbo strongest silver track ", "(0.676 acc) despite 23.6% truncation at 128 tokens."),
         ("Yoruba false-positive skew ", "(101 FP vs 29 FN): MT erases PAWS-style distinctions — 11 identical-text pairs carry contradictory labels."),
         ("Hausa errors are adversarial: ", "44.8% of false positives are ≥0.9-overlap entity/role swaps."),
         ("Yoruba silver gain over TF-IDF ", "(+4.3) is not significant at any seed (min-p 0.066) — reported, not hidden.")],
        size=10.5, gap=6)
footer(s, 8)

# ------------------------------------------------- Slide 9: prototype
s = add_slide()
header(s, "System Prototype, Application and Testing",
       "Live Streamlit application serving the fine-tuned seed-42 checkpoints (localhost)")
shots = ["screenshot_2_hausa_paraphrase.png", "screenshot_3_yoruba_not_paraphrase.png"]
caps = ["Hausa (human labels): held-out pair → paraphrase, calibrated 0.927 (τ = 0.37)",
        "Yoruba (silver labels): held-out example → not paraphrase, calibrated 0.381 (τ = 0.48)"]
x = Inches(0.55)
for f, cap in zip(shots, caps):
    s.shapes.add_picture(str(ASSETS / f), x, Inches(1.5), width=Inches(5.95))
    tb(s, x, Inches(5.22), Inches(5.95), Inches(0.4), cap, size=10, color=INK2, align=PP_ALIGN.CENTER)
    x += Inches(6.25)
bullets(s, Inches(0.55), Inches(5.8), Inches(12.2), Inches(1.5),
        [("Testing: ", "72/72 unit tests (metrics, normalisation, splits, calibration, inference); demo verified headless (HTTP 200)."),
         ("Consistency: ", "the app uses the same clean_text, dev-tuned thresholds and dev-fitted temperatures as the reported experiments — no hardcoded predictions."),
         ("Robustness: ", "input validation (both sentences required), model caching (load ≈ 10 s, inference < 1 s), provenance warnings for silver-trained languages.")],
        size=11, gap=6)
footer(s, 9)

# ------------------------------- Slide 10: limitations/conclusion/future
s = add_slide()
header(s, "Limitations, Conclusion and Recommendations")
panel(s, Inches(0.5), Inches(1.45), Inches(4.05), Inches(3.9))
panel(s, Inches(4.72), Inches(1.45), Inches(4.05), Inches(3.9))
panel(s, Inches(8.94), Inches(1.45), Inches(3.9), Inches(3.9))
bullets(s, Inches(0.68), Inches(1.62), Inches(3.7), Inches(3.6),
        [("Silver labels unverified: ", "gold-validation tooling built (390 pairs) but skipped by decision."),
         ("Truncation: ", "44.8% Yoruba / 23.6% Igbo pairs cut at 128 tokens."),
         ("Small silver test folds ", "(148–295 pairs) → wide CIs."),
         ("SemRel/IGBONLP licences unstated ", "(research use only)."),
         ("7 embedding runs needed ", "OOM re-runs (reduced batch).")],
        title="Limitations", size=10.5, gap=6)
bullets(s, Inches(4.90), Inches(1.62), Inches(3.7), Inches(3.6),
        [("Problem: ", "no paraphrase data / leaky splits for yo–ig–ha."),
         ("Approach: ", "constructed datasets + 29-run transformer benchmark + exact significance testing."),
         ("System: ", "calibrated Streamlit demo, 3 languages."),
         ("Key findings: ", "0.783 acc / ρ 0.697 on clean Hausa data; leakage inflates ≈ +6 pts and flips model selection; label provenance outweighs architecture (ρ 0.153 silver vs 0.697).")],
        title="Conclusion", size=10.5, gap=6)
bullets(s, Inches(9.12), Inches(1.62), Inches(3.55), Inches(3.6),
        [("Run the built gold validation ", "(390 Yoruba candidates)."),
         ("Retrain Yoruba/Igbo at maxlen 256", " to remove truncation."),
         ("Filter NLLB degeneration loops ", "(2.3% of Yoruba pairs)."),
         ("Native-speaker translation of PAWS ", "(PAWS-X route)."),
         ("Extend to more African languages; ", "distil for on-device serving.")],
        title="Recommendations (future)", size=10.5, gap=6)
tb(s, Inches(0.5), Inches(5.55), Inches(12.33), Inches(1.1),
   "[1] J. Alabi et al., “Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning,” COLING 2022.   "
   "[2] K. Ogueji, Y. Zhu, J. Lin, “Small Data? No Problem!…,” MRL @ EMNLP 2021.   "
   "[3] F. Feng et al., “Language-agnostic BERT Sentence Embedding (LaBSE),” ACL 2022.   "
   "[4] N. Ousidhoum et al., “SemRel2024 / SemEval-2024 Task 1,” Findings of ACL 2024.   "
   "[5] Y. Yang et al., “PAWS-X,” EMNLP-IJCNLP 2019.   "
   "[6] C. Guo et al., “On Calibration of Modern Neural Networks,” ICML 2017.   "
   "[7] T. G. Dietterich, “Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms,” Neural Computation, 1998.",
   size=8.5, color=INK2)
tb(s, Inches(0.5), Inches(6.75), Inches(12.33), Inches(0.5), "THANK YOU",
   size=20, bold=True, color=CYAN, align=PP_ALIGN.CENTER)
footer(s, 10)

out = HERE / "deep_learning_project_presentation.pptx"
prs.save(str(out))
print(f"wrote {out} with {len(prs.slides._sldIdLst)} slides")

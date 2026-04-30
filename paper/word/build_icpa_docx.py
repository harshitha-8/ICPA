#!/usr/bin/env python3
"""Build the Word-first ICPA manuscript skeleton."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path(__file__).with_name("icpa_manuscript_draft.docx")


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = Pt(9)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_fixed_width(table, widths) -> None:
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    tbl_grid = table._tbl.tblGrid
    if tbl_grid is None:
        tbl_grid = OxmlElement("w:tblGrid")
        table._tbl.insert(0, tbl_grid)
    for child in list(tbl_grid):
        tbl_grid.remove(child)
    for width in widths:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(int(width * 1440)))
        tbl_grid.append(grid_col)
    for row in table.rows:
        for idx, width in enumerate(widths):
            cell = row.cells[idx]
            cell.width = Inches(width)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 1440)))
            tc_w.set(qn("w:type"), "dxa")


def add_caption(doc: Document, label: str, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(9)
    run = p.add_run(label)
    run.bold = True
    p.add_run(" " + text)


def add_note(doc: Document, title: str, body: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.right_indent = Inches(0.2)
    r = p.add_run(title + ": ")
    r.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(31, 78, 121)
    b = p.add_run(body)
    b.font.size = Pt(10)


def add_table(doc: Document, caption_label: str, caption: str, headers, rows, widths=None) -> None:
    # Artifact-tool currently renders generated Word tables poorly in this
    # environment. Keep table shells as clean manuscript text; replace these
    # with native Word tables inside the official ICPA template before final upload.
    add_caption(doc, caption_label, caption)
    header_p = doc.add_paragraph()
    header_p.paragraph_format.left_indent = Inches(0.2)
    header_run = header_p.add_run("Columns: " + " | ".join(headers))
    header_run.bold = True
    header_run.font.size = Pt(9)
    for row in rows:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(" | ".join(str(v) for v in row))
        run.font.size = Pt(9)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(11)
    styles["Normal"].paragraph_format.line_spacing = 1.0
    styles["Normal"].paragraph_format.space_after = Pt(6)

    for name, size, color in [
        ("Title", 16, RGBColor(0, 0, 0)),
        ("Heading 1", 13, RGBColor(31, 78, 121)),
        ("Heading 2", 11, RGBColor(31, 78, 121)),
        ("Heading 3", 10, RGBColor(31, 78, 121)),
    ]:
        style = styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = color
        style.font.bold = True


def add_front_matter(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run(
        "Detection-Guided Semantic 3D Phenotyping of Cotton Bolls from UAV Imagery Before and After Defoliation"
    )
    r.bold = True
    r.font.size = Pt(16)

    authors = doc.add_paragraph()
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    authors.add_run("Harshitha Manjunatha; co-authors to be added").font.size = Pt(11)

    affiliation = doc.add_paragraph()
    affiliation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    affiliation.add_run("Affiliations and corresponding-author email to be added").italic = True

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "Cotton boll phenotyping from unmanned aerial vehicle (UAV) imagery is often treated as a two-dimensional "
        "detection or counting problem, although agronomic decisions often depend on organ-level morphology and "
        "visibility. This manuscript develops a detection-guided 3D phenotyping framework that uses weak boll detections, "
        "mask refinement, camera geometry, semantic feature correspondence, and multi-view association to estimate "
        "boll count, 3D location, diameter, volume, visibility, and occlusion before and after defoliation. The paired "
        "pre- and post-defoliation flights provide a controlled visibility intervention for quantifying how defoliation "
        "changes recoverability and measurement stability. Final experiments will report correspondence quality, "
        "reconstruction quality, boll recovery, morphology accuracy, and robustness across reconstruction and association settings."
    )
    doc.add_paragraph(
        "Keywords: precision agriculture; cotton phenotyping; UAV imagery; 3D reconstruction; boll morphology; "
        "semantic correspondence; defoliation"
    )


def add_section_plan(doc: Document) -> None:
    doc.add_heading("1 Introduction", level=1)
    doc.add_paragraph(
        "This section should motivate why cotton boll analysis must move beyond 2D counting. It should introduce "
        "defoliation as a visibility intervention, explain why textureless cotton lint challenges classical reconstruction, "
        "and state the exact contributions: detection-guided 3D localization, morphology estimation, pre/post evaluation, "
        "and robustness analysis."
    )
    add_note(
        doc,
        "Writing target",
        "Use objective, past-tense technical prose. Avoid claiming perfect field-scale reconstruction; claim measured high-confidence morphology."
    )

    doc.add_heading("2 Related Work", level=1)
    doc.add_paragraph(
        "Organize this section around four threads: cotton boll detection and yield estimation; 3D plant phenotyping; "
        "foundation-model features for correspondence; and promptable segmentation. The closest prior work must be "
        "treated directly, especially UAV-based cotton boll 3D reconstruction and Cotton3DGaussians."
    )

    doc.add_heading("3 Method", level=1)
    doc.add_paragraph(
        "The method section should be modular: dataset audit, 2D detector prior, mask refinement, camera geometry, "
        "DINOv2 correspondence, multi-view association, 3D localization, and morphology extraction. The LLM/reporting "
        "component should remain optional and downstream."
    )
    add_table(
        doc,
        "Table 1",
        "Architecture modules and expected outputs. This table should stay early in the method section to keep the system readable.",
        ["Module", "Input", "Output", "Purpose"],
        [
            ["Dataset audit", "Raw UAV folders", "Clean manifest", "Prevent leakage and duplicate-frame confusion"],
            ["2D detection", "RGB image", "Boxes, centers, count prior", "Reuse prior accepted counting work"],
            ["Mask refinement", "Boxes/centers", "Approx. silhouettes", "Support geometry and size estimation"],
            ["Geometry", "Image sequence", "Camera poses, point cloud", "Establish metric 3D frame"],
            ["Association", "Detections, poses, features", "Boll tracks", "Link same boll across views"],
            ["Morphology", "3D tracks", "Diameter, volume, visibility", "Deliver phenotyping measurements"],
        ],
        widths=[1.05, 1.25, 1.35, 2.35],
    )

    doc.add_heading("4 Experiments", level=1)
    doc.add_paragraph(
        "The experiments should resemble a strong computer-vision paper: main result, robustness grid, component "
        "ablation, pre/post defoliation comparison, qualitative reconstruction figure, and failure analysis."
    )
    add_table(
        doc,
        "Table 2",
        "Main reconstruction and boll recovery comparison. Replace TBD values only with measured results.",
        ["Category", "Method", "Registered", "Points (K)", "RC up", "BRR up"],
        [
            ["Classical", "COLMAP-SIFT Pre", "TBD", "TBD", "TBD", "TBD"],
            ["Classical", "COLMAP-SIFT Post", "TBD", "TBD", "TBD", "TBD"],
            ["Foundation", "DUSt3R/MASt3R subset", "TBD", "TBD", "TBD", "TBD"],
            ["Ours", "Detection + COLMAP", "TBD", "TBD", "TBD", "TBD"],
            ["Ours", "Detection + COLMAP + DINOv2", "TBD", "TBD", "TBD", "TBD"],
        ],
        widths=[0.75, 1.75, 0.75, 0.75, 0.65, 0.65],
    )
    add_table(
        doc,
        "Table 3",
        "Robustness grid for detection-guided 3D boll phenotyping.",
        ["Block", "Variant", "BRR up", "Diam. MAE", "Vol. err.", "Visibility"],
        [
            ["Detector", "Classical prior", "TBD", "TBD", "TBD", "TBD"],
            ["Detector", "Prior + SAM prompt", "TBD", "TBD", "TBD", "TBD"],
            ["Features", "SIFT only", "TBD", "TBD", "TBD", "TBD"],
            ["Features", "DINOv2-S/14", "TBD", "TBD", "TBD", "TBD"],
            ["Features", "DINOv2-B/14", "TBD", "TBD", "TBD", "TBD"],
            ["Features", "DINOv2-L/14", "TBD", "TBD", "TBD", "TBD"],
            ["Association", "Geometry only", "TBD", "TBD", "TBD", "TBD"],
            ["Association", "Geometry + semantic + mask", "TBD", "TBD", "TBD", "TBD"],
        ],
        widths=[0.75, 1.8, 0.65, 0.75, 0.75, 0.75],
    )
    add_table(
        doc,
        "Table 4",
        "Pre- and post-defoliation morphology statistics.",
        ["Trait", "Pre mean +/- SD", "Post mean +/- SD", "Delta", "Effect", "p"],
        [
            ["Boll count", "TBD", "TBD", "TBD", "TBD", "TBD"],
            ["3D recovered bolls", "TBD", "TBD", "TBD", "TBD", "TBD"],
            ["Diameter (mm)", "TBD", "TBD", "TBD", "TBD", "TBD"],
            ["Volume (mm3)", "TBD", "TBD", "TBD", "TBD", "TBD"],
            ["Visibility", "TBD", "TBD", "TBD", "TBD", "TBD"],
            ["Occlusion", "TBD", "TBD", "TBD", "TBD", "TBD"],
        ],
        widths=[1.15, 1.1, 1.1, 0.65, 0.65, 0.45],
    )

    doc.add_heading("5 Discussion", level=1)
    doc.add_paragraph(
        "Discuss what was reliable, what failed, and what must be treated as high-confidence subset morphology rather "
        "than whole-field perfect reconstruction. This is where duplicate frames, wind, weak baselines, scale uncertainty, "
        "and repeated white lint texture should be handled frankly."
    )

    doc.add_heading("6 Conclusion", level=1)
    doc.add_paragraph(
        "Conclude with the verified contribution: detection-guided semantic 3D phenotyping of cotton bolls, evaluated "
        "before and after defoliation, with measured count, location, diameter, volume, visibility, and occlusion."
    )

    doc.add_heading("References", level=1)
    doc.add_paragraph(
        "Use author-year references following the ICPA/Precision Agriculture guidance. Include only cited and verified published or accepted works."
    )
    add_note(
        doc,
        "ICPA format checkpoint",
        "The 2026 full-paper instruction PDF requires Microsoft Word, camera-ready inline figures and tables, a maximum of 15 pages including references, and a conservative manuscript deadline of 22 May 2026."
    )


def main() -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    configure_styles(doc)
    add_front_matter(doc)
    add_section_plan(doc)
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

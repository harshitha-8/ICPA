#!/usr/bin/env python3
"""Build the Word-first ICPA 2026 paper skeleton.

The official ICPA full-paper instructions require a Microsoft Word manuscript.
This builder keeps a reproducible source for the working skeleton and exports
both a DOCX draft and a Markdown planning copy.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


BASE = Path(__file__).resolve().parent
DOCX_OUT = BASE / "icpa_2026_mask_guided_cotton_skeleton.docx"
MD_OUT = BASE / "icpa_2026_mask_guided_cotton_skeleton.md"


TITLE = "Mask-Guided 3D Cotton Boll Reconstruction for Pre- and Post-Defoliation Phenotyping"


ABSTRACT = (
    "Accurate cotton boll phenotyping remains difficult in field imagery because the traits most relevant to yield "
    "and maturity are expressed at the organ scale, while unmanned aerial vehicle (UAV) observations are commonly "
    "summarized as two-dimensional counts or canopy-level indices. This paper skeleton develops a Word-first ICPA "
    "manuscript around a mask-guided reconstruction pipeline for paired pre- and post-defoliation cotton imagery. "
    "The proposed system begins with phase-aware cotton boll candidate detection, refines high-confidence candidates "
    "with SAM-style lint masks, projects mask-selected pixels into a morphology-aware 3D review space, and aggregates "
    "boll count, visibility, mask length, mask width, diameter, and ellipsoid volume proxies at both image and plot-cell "
    "levels. The paired pre/post setting is framed as a controlled visibility intervention rather than a conventional "
    "train/test split: post-defoliation imagery is expected to expose bolls that are partially or fully occluded in "
    "pre-defoliation imagery, enabling direct analysis of recoverability and measurement stability. The current system "
    "is intentionally described as a proxy pipeline until metric scale is validated through ground sampling distance, "
    "camera calibration, ground control, or direct physical measurements. The planned evaluation compares detection, "
    "segmentation, mask-to-3D review, trait estimation, plot mapping, robustness, and optional agronomic reporting "
    "modules without inventing unsupported performance claims. The intended contribution is a practical bridge from "
    "UAV cotton boll counting toward calibrated organ-scale 3D phenotyping."
)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.space_after = Pt(4)

    for style_name, size, color in [
        ("Title", 16, RGBColor(0, 0, 0)),
        ("Heading 1", 12.5, RGBColor(31, 78, 121)),
        ("Heading 2", 11, RGBColor(31, 78, 121)),
        ("Heading 3", 10.5, RGBColor(31, 78, 121)),
    ]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color

    header = section.header.paragraphs[0]
    header.text = "ICPA 2026 working manuscript skeleton"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(100, 100, 100)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.text = "Draft skeleton; replace placeholders with verified results."
    for run in footer.runs:
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(100, 100, 100)


def add_paragraph(doc: Document, text: str = "", style: str | None = None, italic: bool = False) -> None:
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.italic = italic


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.add_run(item)


def add_numbered(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(3)
        p.add_run(item)


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell(cell, text: str, bold: bool = False, align: str = "left") -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(8.8)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_widths(table, widths_in: list[float]) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
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
    for width in widths_in:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(int(width * 1440)))
        tbl_grid.append(grid_col)

    for row in table.rows:
        for idx, width in enumerate(widths_in):
            cell = row.cells[idx]
            cell.width = Inches(width)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 1440)))
            tc_w.set(qn("w:type"), "dxa")


def add_table(doc: Document, caption: str, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    label, rest = caption.split(" ", 1)
    r = p.add_run(label + " ")
    r.bold = True
    p.add_run(rest)

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_widths(table, widths)
    for idx, head in enumerate(headers):
        shade_cell(table.rows[0].cells[idx], "EAF1E8")
        set_cell(table.rows[0].cells[idx], head, bold=True, align="center")
    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            set_cell(cells[idx], value, align="center" if len(value) < 18 else "left")


def add_algorithm(doc: Document, title: str, steps: list[str]) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(title)
    run.bold = True
    for step in steps:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.left_indent = Inches(0.28)
        p.paragraph_format.space_after = Pt(2)
        p.add_run(step)


def add_equations(doc: Document) -> None:
    add_paragraph(doc, "Equation placeholders to number sequentially in the final manuscript:")
    equations = [
        "(1)  L_mm = l_px * GSD_mm_per_px",
        "(2)  W_mm = w_px * GSD_mm_per_px",
        "(3)  V_proxy = (4/3) * pi * (L/2) * (W/2) * (W/2)",
        "(4)  visibility = contour_area / bounding_box_area",
        "(5)  confidence = f(lint fraction, visibility, depth, size prior, brightness, green penalty)",
    ]
    for eq in equations:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(eq)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)


def build_docx() -> None:
    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run(TITLE)
    add_paragraph(doc, "Author 1; Author 2; Author 3", italic=False)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_paragraph(doc, "Affiliations, addresses, and corresponding-author email to be inserted.", italic=True)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_paragraph(doc, "Keywords: precision agriculture; cotton phenotyping; UAV imagery; 3D reconstruction; promptable segmentation; defoliation")

    doc.add_heading("Abstract", level=1)
    add_paragraph(doc, ABSTRACT)

    doc.add_heading("1 Introduction", level=1)
    add_paragraph(doc, "Cotton boll phenotyping is central to yield estimation, harvest timing, defoliation assessment, and breeding decisions, yet UAV pipelines usually compress boll evidence into two-dimensional counts. The introduction should open with this agronomic need and then show why count alone is incomplete: boll size, visibility, occlusion, and spatial distribution determine whether a count is interpretable.")
    add_paragraph(doc, "The paired pre- and post-defoliation setting should be introduced as the paper's controlled visibility intervention. The argument is not simply that there are two folders of images; it is that defoliation changes the observable surface of the crop and therefore allows an explicit test of what organ-scale phenotyping gains when occluding foliage is reduced.")
    add_paragraph(doc, "The final paragraphs should motivate mask-guided 3D review. The current manuscript should avoid claiming solved metric 3D. Instead, it should argue that detector-prompted masks, proxy depth, and highlighted 3D review create a practical bridge toward calibrated boll-scale reconstruction.")
    add_bullets(doc, [
        "Contribution 1: a pre/post-defoliation UAV study design for visibility-aware cotton boll phenotyping.",
        "Contribution 2: a mask-guided boll extraction pipeline that extends counting into measurement-ready candidates.",
        "Contribution 3: a mask-to-3D review module that exports a boll-mask point cloud and supports visual inspection of organ-scale reconstruction candidates.",
        "Contribution 4: proxy trait definitions for count, visibility, mask length, mask width, diameter, ellipsoid volume, extraction confidence, and plot-cell aggregation.",
        "Contribution 5: an evaluation plan that separates reconstruction evidence from optional agronomic reporting or LLM-based decision support.",
    ])

    doc.add_heading("2 Related Work", level=1)
    for heading, body, gap in [
        ("2.1 UAV and field-based cotton phenotyping", "Cite precision-agriculture work that estimates cotton stand count, canopy cover, plant height, yield proxies, and defoliation effects from UAV or field imaging.", "Gap: most pipelines remain canopy- or count-oriented and do not validate organ-scale boll morphology under paired pre/post visibility changes."),
        ("2.2 Cotton boll detection and counting", "Cite classical image processing, deep detectors, detector-fusion, and prior accepted work from this project on cotton boll counting.", "Gap: detection boxes are rarely converted into measurement-oriented masks, 3D review objects, or trait uncertainty estimates."),
        ("2.3 3D plant reconstruction and point-cloud phenotyping", "Cite plant point-cloud reconstruction, row/plot segmentation, soil removal, cotton boll point-cloud counting, and 3D morphology estimation.", "Gap: field-scale point clouds can support boll traits, but dense manual cleanup, calibration, and occlusion remain limiting factors."),
        ("2.4 Segmentation foundation models", "Cite Segment Anything, SAM2, and promptable segmentation papers, plus agricultural segmentation adaptations where available.", "Gap: promptable masks are useful, but their role in UAV cotton boll measurement requires validation against expert masks and physical traits."),
        ("2.5 Gaussian Splatting, NeRF, and novel-view reconstruction", "Cite NeRF, 3D Gaussian Splatting, agricultural Gaussian Splatting work, and any cotton-specific 3DGS references actually used.", "Gap: photorealistic reconstruction is not equivalent to calibrated organ measurement; the paper must test trait accuracy rather than visual plausibility alone."),
        ("2.6 Agronomic decision support", "If retained, cite LLM or multimodal reporting work only for structured interpretation of measured traits.", "Gap: LLMs should be evaluated for faithfulness and schema validity, not treated as a geometry engine."),
    ]:
        doc.add_heading(heading, level=2)
        add_paragraph(doc, body)
        add_paragraph(doc, gap, italic=True)

    doc.add_heading("3 Dataset and Study Setting", level=1)
    add_paragraph(doc, "Dataset folders should be reported exactly and audited before final submission: pre-defoliation images are stored in Part_one_pre_def_rgb and part 2_pre_def_rgb; post-defoliation images are stored in 205_Post_Def_rgb, Post_def_rgb_part1, part3_post_def_rgb, and part4_post_def_rgb.")
    add_bullets(doc, [
        "Insert acquisition metadata: UAV model, camera model, flight altitude, overlap, date, time, field location, and weather/wind if available.",
        "Insert calibration metadata: camera intrinsics, ground sampling distance, ground control points, GPS/RTK quality, plot boundary coordinates, and orthomosaic availability.",
        "State available supervision: prior detector outputs, pseudo-labels, manual count checks, expert masks, physical boll measurements, or absence of each.",
        "Explain annotation cost: dense boll annotation can take several days per high-resolution image, motivating weak prompts and high-confidence subsets.",
        "State dataset limitations: scale ambiguity, repeated white lint texture, overlapping bolls, residue/soil false positives, image duplicates, and occlusion.",
    ])
    add_table(doc, "Table 1 Dataset statistics to insert after audit.", ["Split", "Folders", "Frames", "Labels", "Purpose"], [
        ["Pre-defoliation", "Part_one_pre_def_rgb; part 2_pre_def_rgb", "TBD", "TBD", "Canopy occlusion and baseline visibility"],
        ["Post-defoliation", "205_Post_Def_rgb; Post_def_rgb_part1; part3_post_def_rgb; part4_post_def_rgb", "TBD", "TBD", "Reduced foliage and boll recoverability"],
        ["Validation subset", "Manually selected frames", "TBD", "Counts/masks/traits TBD", "Metric and failure analysis"],
    ], [1.05, 2.25, 0.55, 1.25, 1.4])

    doc.add_heading("4 Method", level=1)
    doc.add_heading("4.1 Overview", level=2)
    add_paragraph(doc, "Pipeline: UAV frame -> phase identification -> boll candidate detection -> SAM-style mask extraction -> morphology-depth proxy -> mask-to-3D projection -> trait estimation -> plot-cell aggregation -> evaluation/reporting. Fig. 1 should present this as a left-to-right architecture with the decision loop separated from geometry.")
    doc.add_heading("4.2 Cotton boll candidate detection", level=2)
    add_paragraph(doc, "Describe the current detector: color normalization with CLAHE, multi-scale top-hat morphology for bright lint structures, Otsu thresholding, contour extraction, saturation/value gates, aspect-ratio filtering, and phase-dependent count adjustment. This section should cite the earlier accepted counting work and present the detector as a reusable prior, not the sole contribution.")
    doc.add_heading("4.3 SAM-style boll mask extraction", level=2)
    add_paragraph(doc, "Detector boxes serve as prompts. Within each prompted crop, low-saturation and high-value lint pixels are retained, morphological cleanup is applied, and the largest connected lint component is selected as the candidate mask. The official SAM/SAM2 model can replace this deterministic slot; the manuscript should compare it only after inference and validation are actually run.")
    doc.add_heading("4.4 Proxy 3D reconstruction and mask-to-3D review", level=2)
    add_paragraph(doc, "The MVP uses a morphology-aware depth proxy and exports both a scene point cloud and a boll-mask point cloud. Mask pixels are projected into the same 3D review coordinate system, allowing the full image scene to be inspected with highlighted boll evidence. For final metric claims, replace or validate this stage with calibrated SfM, COLMAP, 3DGS, NeRF, RGB-D, GCP-supported orthomosaic geometry, or direct scale references.")
    doc.add_heading("4.5 Trait estimation", level=2)
    add_paragraph(doc, "Define count, visibility, mask area, mask length, mask width, bounding-box diameter, ellipsoid volume proxy, extraction confidence, and plot-cell summaries. The final manuscript should use SI units; report mm and mm3 after scale validation rather than cm and cm3.")
    add_equations(doc)
    doc.add_heading("4.6 Plot-level mapping", level=2)
    add_paragraph(doc, "The current app overlays a 4 x 43 plot-grid proxy on the central study region and assigns measurement-ready candidates to image-coordinate cells. This is appropriate for method development and visual quality control. It becomes a metric field map only after orthomosaic, camera pose, GCP, or plot-boundary calibration is added.")
    doc.add_heading("4.7 Agronomist-in-the-loop decision layer", level=2)
    add_paragraph(doc, "This layer should be optional and downstream. It receives measured morphology JSON and produces structured summaries or recommendations. It must not be described as performing reconstruction. Evaluate schema validity, expert agreement, hallucination rate, latency, and recommendation consistency across two or three open-source models if included.")

    doc.add_heading("5 Algorithms", level=1)
    add_algorithm(doc, "Algorithm 1. Mask-guided cotton boll phenotyping pipeline.", [
        "Input UAV frame, phase label or phase estimator, scale assumption or calibration metadata.",
        "Detect cotton boll candidates using the phase-aware detector or a learned detector baseline.",
        "For each candidate, extract a prompt-guided lint mask and retain the largest plausible component.",
        "Estimate mask shape, visibility, confidence, and proxy traits.",
        "Project selected mask pixels into the reconstruction/review coordinate system.",
        "Aggregate candidates by plot cell and export scene PLY, boll-mask PLY, CSV, and figures.",
    ])
    add_algorithm(doc, "Algorithm 2. SAM-style boll mask extraction.", [
        "Crop a padded candidate region around each detector box.",
        "Compute color gates for low-saturation, high-value cotton lint while suppressing green canopy.",
        "Apply morphological open/close operations to remove isolated noise.",
        "Select the largest connected lint component as the candidate mask.",
        "Return mask area, oriented length, oriented width, and mask confidence inputs.",
    ])
    add_algorithm(doc, "Algorithm 3. Mask-to-3D projection and trait estimation.", [
        "Map each selected mask pixel from original image coordinates to the resized depth grid.",
        "Back-project the pixel into normalized scene coordinates using the depth proxy or calibrated geometry.",
        "Color the projected points by candidate identity or confidence.",
        "Export a boll-mask PLY and render highlighted points over the full scene cloud.",
        "Compute length, width, and ellipsoid volume proxies from the mask metrics and scale metadata.",
    ])
    add_algorithm(doc, "Algorithm 4. Plot-cell aggregation.", [
        "Define the plot grid in image coordinates or calibrated field coordinates.",
        "Assign each measurement-ready boll center to a row-column cell.",
        "Compute cell-level count, mean trait values, confidence, and coverage.",
        "Report cells with high count, high uncertainty, or strong pre/post change.",
    ])

    doc.add_heading("6 Experimental Design", level=1)
    add_bullets(doc, [
        "Pre/post comparison: quantify raw detections, measurement-ready candidates, visibility, mask traits, and plot-cell changes between paired phases.",
        "Detector baselines: classical detector, YOLO-family detector if trained, GroundingDINO/OWL-style prompt detector if run, and detector-fusion variants.",
        "Segmentation baselines: box crop thresholding, current SAM-style mask, official SAM/SAM2 prompted masks if integrated, and expert masks on a small validation subset.",
        "Reconstruction baselines: morphology-depth proxy, COLMAP/SfM, SuperPoint/SuperGlue or similar local features, DINOv2 feature matching, 3DGS/NeRF only if actually run.",
        "Ablation study: detector only; detector plus mask; detector plus mask plus depth projection; detector plus mask plus calibrated geometry; optional decision layer.",
        "Robustness study: scale, occlusion, brightness, canopy density, row position, image blur, and pre/post phase.",
        "Runtime study: detector time, mask extraction time, point-cloud export time, and reporting-layer latency.",
        "Failure analysis: bright residue, soil, overlapping bolls, leaf glare, low texture, repeated lint patterns, and weak scale calibration.",
    ])

    doc.add_heading("7 Evaluation Metrics", level=1)
    add_table(doc, "Table 2 Evaluation metric plan.", ["Module", "Metrics", "Evidence required"], [
        ["Detection", "Precision; recall; F1; MAE count; RMSE count", "Manual boxes/counts on validation frames"],
        ["Segmentation", "Mask IoU; boundary F1; mask area error", "Expert masks or audited subset"],
        ["3D/reconstruction", "Chamfer distance; completeness; point density; reprojection consistency; rendered-view quality", "Ground truth or calibrated reference where available"],
        ["Trait estimation", "Length MAE; width MAE; volume error; correlation", "Physical boll measurements or trusted scale references"],
        ["Plot mapping", "Cell count error; row/column consistency; pre/post change stability", "Plot boundaries and manual cell checks"],
        ["Decision loop", "Schema validity; expert alignment; hallucination rate; latency", "Expert rubric and structured output schema"],
    ], [1.15, 2.3, 2.55])

    doc.add_heading("8 Results Skeleton", level=1)
    add_table(doc, "Table 3 Results placeholders. Do not replace TBD until experiments are run.", ["Table", "Purpose", "Required content"], [
        ["Table 1", "Dataset statistics", "Frame counts, phase split, validation subset, labels"],
        ["Table 2", "Detection and counting performance", "Precision, recall, F1, MAE, RMSE"],
        ["Table 3", "Mask extraction ablation", "Box-only vs SAM-style vs SAM/SAM2 if run"],
        ["Table 4", "3D reconstruction review quality", "Point density, completeness, reprojection consistency"],
        ["Table 5", "Trait estimation accuracy", "Length, width, diameter, volume errors"],
        ["Table 6", "Pre vs post comparison", "Visibility, recoverability, count and trait deltas"],
        ["Table 7", "Robustness", "Scale, occlusion, brightness, canopy density"],
        ["Table 8", "Decision-loop comparison", "Only if LLM reporting is included"],
    ], [0.75, 1.55, 3.7])

    doc.add_heading("9 Figure Plan", level=1)
    add_numbered(doc, [
        "Overall architecture: show the reconstruction path separately from the optional reporting loop.",
        "Pre/post defoliation examples: paired imagery with visible changes in foliage and boll exposure.",
        "Detector to mask to mask-to-3D projection: show one candidate through all stages.",
        "SAM-style mask overlay: translucent masks on the real cotton image, not synthetic blocks.",
        "Interactive highlighted boll point cloud: full scene with colored boll-mask points.",
        "Plot-grid mapping: 4 x 43 proxy grid with cell-level count and confidence.",
        "Trait estimation examples: selected bolls with mask length, width, and volume proxy.",
        "Failure cases: false positives, occlusion, overlapping lint, low-quality reconstruction.",
        "Ablation overview: visual and tabular summary of component contributions.",
    ])

    doc.add_heading("10 Discussion", level=1)
    add_paragraph(doc, "Discuss what the current pipeline already supports: rapid inspection of real UAV frames, measurement-ready candidate filtering, mask overlays, mask-to-3D review, and plot-cell summaries. Then separate these strengths from what still requires validation.")
    add_paragraph(doc, "The central interpretation should be agronomic: post-defoliation may improve boll visibility, but the magnitude of that improvement must be measured rather than assumed. The method should be positioned as a practical step toward calibrated 3D phenotyping, especially where full manual annotation is infeasible.")

    doc.add_heading("11 Limitations", level=1)
    add_bullets(doc, [
        "No metric 3D claim is valid without calibration, GSD, GCPs, camera poses, or physical scale references.",
        "UAV scale ambiguity affects length, width, diameter, and volume proxies.",
        "Occlusion and overlapping bolls can merge masks or fragment one boll into several candidates.",
        "Bright soil, residue, or glare can produce detector false positives.",
        "Mask quality depends on the appearance of cotton lint and may fail under shadow or motion blur.",
        "The ellipsoid volume proxy is a geometric approximation, not a direct physical volume measurement.",
        "The LLM/reporting layer should not be treated as agronomic authority without expert validation.",
    ])

    doc.add_heading("12 Conclusion", level=1)
    add_paragraph(doc, "This paper should conclude that mask-guided UAV analysis can extend cotton boll counting toward organ-scale phenotyping by linking detection, prompt-style lint masks, mask-to-3D review, proxy trait estimation, and plot-level aggregation. The final conclusion should report only measured findings and should preserve the distinction between proxy MVP evidence and calibrated metric reconstruction.")

    doc.add_heading("13 Supplementary Material Plan", level=1)
    add_bullets(doc, [
        "Implementation details and parameter tables for detection, masking, depth proxy, and plot mapping.",
        "Additional pre/post visual examples and mask overlays.",
        "Robustness tables for brightness, scale, occlusion, canopy density, and image quality.",
        "Failure-case gallery with explanation of each error mode.",
        "Annotation protocol for count, mask, and physical trait validation.",
        "Prompts, schemas, and scoring rubrics for the optional LLM decision loop.",
        "Dataset split manifest and duplicate-frame audit.",
    ])

    doc.add_heading("14 Patent-Aware Internal Notes - Not for Submission", level=1)
    add_bullets(doc, [
        "Potential system claim: mask-guided UAV cotton boll trait estimation linking candidate detection, promptable segmentation, and 3D review.",
        "Potential workflow claim: pre/post-defoliation visibility intervention for organ-scale recoverability measurement.",
        "Potential interface claim: mask-to-3D boll review with separate boll-mask point-cloud export and plot-cell aggregation.",
        "Do not include legal claims in the submitted manuscript; discuss novelty with a patent professional before disclosure.",
    ])

    doc.add_section(WD_SECTION.CONTINUOUS)
    doc.add_heading("References Placeholder", level=1)
    add_paragraph(doc, "Insert only cited, verified, published, or accepted references. Use author-year citations and alphabetize entries by first-author surname. Journal names and book titles should be italicized in the final Word manuscript.")

    doc.save(DOCX_OUT)


def build_markdown() -> None:
    text = f"""# {TITLE}

Author placeholders: Author 1; Author 2; Author 3

Affiliations and corresponding-author email to be inserted.

Keywords: precision agriculture; cotton phenotyping; UAV imagery; 3D reconstruction; promptable segmentation; defoliation

## Abstract

{ABSTRACT}

## Skeleton Use Notes

This source mirrors the Word skeleton generated by `build_icpa_docx.py`. It follows the verified ICPA constraints in `docs/icpa_author_guidelines.md`: Microsoft Word manuscript, maximum 15 pages including figures/tables/references, camera-ready formatting, no more than three displayed heading levels, author-year references, and SI units. The paper should use mm and mm3 for final measurements after scale validation.

## Core Accuracy Boundary

The current implementation is an MVP/proxy pipeline. It should not claim final metric 3D Gaussian Splatting or calibrated multi-view metrology unless those experiments are run and validated. Length, width, diameter, and volume remain proxy traits until ground sampling distance, camera calibration, GCPs, or physical boll measurements are available.

## Required Paper Structure

1. Title page
2. Abstract
3. Introduction
4. Related work
5. Dataset and study setting
6. Method
7. Algorithms
8. Experimental design
9. Evaluation metrics
10. Results skeleton
11. Figure plan
12. Discussion
13. Limitations
14. Conclusion
15. Supplementary material plan
16. Patent-aware internal notes, not for submission

For the full detailed skeleton, use the generated DOCX file.
"""
    MD_OUT.write_text(text, encoding="utf-8")


def main() -> None:
    build_docx()
    build_markdown()
    print(f"Wrote {DOCX_OUT}")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()

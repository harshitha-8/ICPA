#!/usr/bin/env python3
"""Build the Word-first ICPA manuscript draft through Algorithm 4."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


BASE = Path(__file__).resolve().parent
DOCX_OUT = BASE / "icpa_2026_mask_guided_cotton_until_algorithms.docx"
MD_OUT = BASE / "icpa_2026_mask_guided_cotton_until_algorithms.md"
TITLE = "Mask-Guided 3D Cotton Boll Reconstruction for Pre- and Post-Defoliation Phenotyping"


REFERENCES = [
    "DeTone, D., Malisiewicz, T., and Rabinovich, A. 2018. SuperPoint: Self-supervised interest point detection and description. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops.",
    "Edelsbrunner, H., Kirkpatrick, D., and Seidel, R. 1983. On the shape of a set of points in the plane. IEEE Transactions on Information Theory 29(4):551-559.",
    "Jiang, X., Li, Z., Li, D., Zhang, Y., Hu, J., Lin, F., and Zhou, J. 2025. Cotton3DGaussians: 3D Gaussian-based three-dimensional reconstruction and phenotyping for cotton bolls. The Plant Phenomics.",
    "Kerbl, B., Kopanas, G., Leimkuehler, T., and Drettakis, G. 2023. 3D Gaussian Splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4):1-14.",
    "Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., et al. 2023. Segment Anything. Proceedings of the IEEE/CVF International Conference on Computer Vision.",
    "Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R. 2020. NeRF: Representing scenes as neural radiance fields for view synthesis. Proceedings of the European Conference on Computer Vision.",
    "Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., et al. 2023. DINOv2: Learning robust visual features without supervision. arXiv:2304.07193.",
    "Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., et al. 2024. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714.",
    "Sarlin, P.-E., DeTone, D., Malisiewicz, T., and Rabinovich, A. 2020. SuperGlue: Learning feature matching with graph neural networks. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.",
    "Schoenberger, J. L., and Frahm, J.-M. 2016. Structure-from-Motion revisited. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.",
    "Schoenberger, J. L., Zheng, E., Frahm, J.-M., and Pollefeys, M. 2016. Pixelwise view selection for unstructured multi-view stereo. Proceedings of the European Conference on Computer Vision.",
    "Sun, S., Li, C., Paterson, A. H., Jiang, Y., Xu, R., Robertson, J. S., et al. 2020. Three-dimensional photogrammetric mapping of cotton bolls in situ based on point cloud segmentation and clustering. ISPRS Journal of Photogrammetry and Remote Sensing 160:195-207.",
    "Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., and Revaud, J. 2024. DUSt3R: Geometric 3D vision made easy. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.",
    "Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., and Revaud, J. 2024. MASt3R: Grounding image matching in 3D with mast3r. Proceedings of the European Conference on Computer Vision.",
    "Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., and Novotny, D. 2025. VGGT: Visual Geometry Grounded Transformer. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.",
    "Xiao, S., Fei, S., Ye, Y., Xu, D., Xie, Z., Bi, K., et al. 2024. 3D reconstruction and characterization of cotton bolls in situ based on UAV technology. ISPRS Journal of Photogrammetry and Remote Sensing 209:101-116.",
]


def configure(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    section.header.paragraphs[0].text = ""
    section.footer.paragraphs[0].text = ""

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.space_after = Pt(4)

    for style_name, size in [("Title", 15), ("Heading 1", 12), ("Heading 2", 11), ("Heading 3", 10.5)]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)


def paragraph(doc: Document, text: str = "", style: str | None = None, italic: bool = False) -> None:
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.italic = italic


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        p.add_run(item)


def number_steps(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(2)
        p.add_run(item)


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(8.6)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def widths(table, vals: list[float]) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    table._tbl.tblPr.append(layout)
    grid = table._tbl.tblGrid
    if grid is None:
        grid = OxmlElement("w:tblGrid")
        table._tbl.insert(0, grid)
    for child in list(grid):
        grid.remove(child)
    for val in vals:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(int(val * 1440)))
        grid.append(col)
    for row in table.rows:
        for idx, val in enumerate(vals):
            tc_pr = row.cells[idx]._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(val * 1440)))
            tc_w.set(qn("w:type"), "dxa")


def table(doc: Document, caption: str, headers: list[str], rows: list[list[str]], col_widths: list[float]) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    label, rest = caption.split(" ", 1)
    run = p.add_run(label + " ")
    run.bold = True
    p.add_run(rest)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    widths(t, col_widths)
    for idx, header in enumerate(headers):
        shade(t.rows[0].cells[idx], "EDEDED")
        cell_text(t.rows[0].cells[idx], header, True)
    for row in rows:
        cells = t.add_row().cells
        for idx, value in enumerate(row):
            cell_text(cells[idx], value)


def algorithm(doc: Document, title: str, inputs: str, output: str, steps: list[str]) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.bold = True
    paragraph(doc, f"Input: {inputs}", italic=True)
    paragraph(doc, f"Output: {output}", italic=True)
    number_steps(doc, steps)


def equations(doc: Document) -> None:
    for idx, eq in enumerate(
        [
            "L = l_px * s",
            "W = w_px * s",
            "D = 0.5 * (w_box + h_box) * s",
            "V_ellipsoid = (4/3) pi (L/2)(W/2)(W/2)",
            "v = A_contour / A_box",
            "q = f(r_lint, v, d, r_size, b, 1 - r_green)",
        ],
        start=1,
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        p.add_run(f"({idx})  {eq}")


def build_doc() -> None:
    doc = Document()
    configure(doc)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run(TITLE)
    paragraph(doc, "Author names and affiliations to be inserted after co-author confirmation.")
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph(doc, "Corresponding author: to be inserted.")
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph(doc, "Keywords: cotton phenotyping; UAV imagery; 3D reconstruction; promptable segmentation; defoliation; precision agriculture")

    doc.add_heading("Abstract", level=1)
    paragraph(doc, "Cotton boll phenotyping from unmanned aerial vehicle (UAV) imagery is often reduced to two-dimensional counting, even though agronomic interpretation depends on visibility, occlusion, and organ-scale morphology. This manuscript develops a mask-guided framework for pre- and post-defoliation cotton imagery in which defoliation is treated as a controlled visibility intervention rather than a simple data split. The proposed pipeline detects candidate bolls, refines them with prompt-style lint masks inspired by recent segmentation foundation models, projects mask-selected pixels into a 3D review space, and estimates count, visibility, mask length, mask width, diameter, and ellipsoid volume proxies at image and plot-cell levels. The present draft deliberately distinguishes proxy measurements from calibrated metrology: length, diameter, and volume estimates are reported as provisional unless supported by ground sampling distance, camera calibration, ground control points, or direct physical measurements. The paper is positioned as a practical bridge between field-scale cotton boll counting and calibrated organ-scale 3D phenotyping. The planned evaluation will compare detection, segmentation, mask-to-3D projection, trait estimation, and plot mapping under pre- and post-defoliation conditions, with optional decision-support reporting evaluated separately for faithfulness and schema validity. No unverified accuracy claims are made in this manuscript stage.")

    doc.add_heading("1 Introduction", level=1)
    paragraph(doc, "Cotton yield assessment and harvest management depend on the development, exposure, and spatial distribution of bolls. UAV imagery has made it possible to observe large field areas at high temporal frequency, but most image-based boll analyses remain closer to counting than to measurement. A field can contain thousands of bright lint structures in a single frame, and those structures vary in visibility because foliage, branches, shadows, soil, and neighboring bolls obscure the organ boundary. A count can therefore be useful and still incomplete: it says little about which bolls were measurable, which were partially occluded, and whether organ-scale traits were stable enough to support agronomic decisions.")
    paragraph(doc, "The pre- and post-defoliation setting provides an unusually informative view of this problem. Defoliation is not merely a change in image appearance; it is an agronomic intervention that removes part of the canopy and changes the visibility of the reproductive structures. In a pre-defoliation image, a boll may be detectable only as a small white region between leaves. In a post-defoliation image, the same region may become more separable from the canopy, allowing a better estimate of size and location. Treating these phases as a controlled visibility contrast makes the study more precise than a generic pre/post split.")
    paragraph(doc, "The difficulty is that organ-scale 3D measurement is not automatically obtained from UAV imagery. Classical structure-from-motion pipelines such as COLMAP rely on stable local texture and sufficient view overlap, conditions that are not guaranteed for repeated white cotton lint (Schoenberger and Frahm, 2016; Schoenberger et al., 2016). Neural view synthesis and 3D Gaussian Splatting can produce visually compelling scene representations, but photorealistic rendering alone does not prove accurate boll length or volume (Mildenhall et al., 2020; Kerbl et al., 2023). Recent foundation models for segmentation and visual geometry, including Segment Anything, SAM 2, DUSt3R, MASt3R, and VGGT, suggest useful components for promptable masks and geometry-aware reconstruction, but they must be adapted carefully to agricultural imagery (Kirillov et al., 2023; Ravi et al., 2024; Wang et al., 2024; Wang et al., 2025).")
    paragraph(doc, "This paper therefore frames the contribution as a measurement-ready pipeline rather than a solved metrology system. Candidate bolls are detected from UAV frames, refined into lint masks, projected into a proxy 3D review space, and aggregated into plot-level summaries. The system is designed to expose where measurements are reliable and where calibration or additional views are still required.")
    bullets(doc, [
        "A pre/post-defoliation formulation that treats defoliation as a visibility intervention for cotton boll phenotyping.",
        "A detector-prompted mask extraction stage that turns raw boll detections into measurement-ready lint candidates.",
        "A mask-to-3D review module that exports a highlighted boll-mask point cloud and supports visual inspection of organ-level evidence.",
        "Explicit proxy trait definitions for count, visibility, mask length, mask width, diameter, ellipsoid volume, extraction confidence, and plot-cell aggregation.",
    ])

    doc.add_heading("2 Related Work", level=1)
    doc.add_heading("2.1 UAV and field-based cotton phenotyping", level=2)
    paragraph(doc, "UAV phenotyping has become central in precision agriculture because it offers field-scale coverage with repeatable image acquisition. In cotton, UAV and close-range imagery have been used for stand assessment, canopy characterization, boll detection, and yield-related traits. The closest field-scale works for this manuscript are the cotton boll point-cloud studies by Sun et al. (2020) and Xiao et al. (2024), which demonstrate that 3D reconstruction can support boll counting, spatial distribution, volume analysis, and yield estimation when capture geometry is sufficiently stable. The present paper should position its contribution around paired defoliation, mask-guided measurement readiness, and explicit proxy boundaries rather than claiming novelty for cotton 3D reconstruction itself.")
    doc.add_heading("2.2 Cotton boll detection and counting", level=2)
    paragraph(doc, "Cotton boll counting has been studied with classical image processing, supervised object detectors, and fusion strategies. The current project inherits a phase-aware detector based on contrast enhancement, multi-scale top-hat morphology, thresholding, contour filtering, and color gates. This detector is useful because it produces candidate regions without dense manual annotation, but a detector box is not a physical measurement. The gap addressed here is the conversion of detection evidence into masks, proxy traits, and 3D review objects.")
    doc.add_heading("2.3 3D reconstruction and geometry", level=2)
    paragraph(doc, "Structure-from-motion and multi-view stereo remain standard tools for image-based 3D reconstruction (Schoenberger and Frahm, 2016; Schoenberger et al., 2016). Learned local features and matchers, including SuperPoint and SuperGlue, improve correspondence in many settings but still require texture and viewpoint consistency (DeTone et al., 2018; Sarlin et al., 2020). Newer geometry models such as DUSt3R, MASt3R, and VGGT reduce some of the engineering burden by predicting geometric relationships more directly, but their use in dense cotton scenes must be validated rather than assumed (Wang et al., 2024; Wang et al., 2025).")
    doc.add_heading("2.4 Segmentation foundation models", level=2)
    paragraph(doc, "Segment Anything introduced promptable segmentation at large scale and made point-, box-, and mask-conditioned extraction a practical design pattern for downstream systems (Kirillov et al., 2023). SAM 2 extended this idea to images and video, making temporal or sequential mask propagation more accessible (Ravi et al., 2024). In this paper, the term SAM-style refers to the prompt-first design: a detector box identifies a candidate, and a mask stage isolates lint-like pixels. The current implementation does not claim to run official SAM/SAM 2 unless those models are integrated and evaluated.")
    doc.add_heading("2.5 Neural rendering and Gaussian Splatting", level=2)
    paragraph(doc, "NeRF and 3D Gaussian Splatting have changed how scenes are represented for novel-view synthesis (Mildenhall et al., 2020; Kerbl et al., 2023). Cotton3DGaussians is particularly relevant because it connects Gaussian scene representation with cotton boll phenotyping (Jiang et al., 2025). The distinction for this manuscript is scale and evidence: the present work starts from UAV field imagery and paired defoliation, and it treats mask-to-3D review as a measurement-support layer rather than a rendering-only objective.")

    table(doc, "Table 1 Closest-prior positioning for the current manuscript.", ["Topic", "Representative source", "What it contributes", "Remaining gap"], [
        ["SfM/MVS", "Schoenberger and Frahm (2016); Schoenberger et al. (2016)", "Classical camera pose and dense reconstruction", "Cotton lint can be low-texture and repetitive"],
        ["Promptable masks", "Kirillov et al. (2023); Ravi et al. (2024)", "Box/point/video-conditioned segmentation", "Agricultural mask accuracy needs validation"],
        ["Neural 3D", "Mildenhall et al. (2020); Kerbl et al. (2023)", "Novel-view scene representations", "Rendering quality is not trait accuracy"],
        ["Cotton 3D", "Sun et al. (2020); Xiao et al. (2024); Jiang et al. (2025)", "Point-cloud and Gaussian cotton boll phenotyping", "Pre/post defoliation and mask-to-3D review remain distinct"],
    ], [1.0, 1.7, 1.8, 2.0])

    doc.add_heading("3 Dataset and Study Setting", level=1)
    paragraph(doc, "The study uses UAV imagery collected before and after defoliation. Pre-defoliation frames are stored in Part_one_pre_def_rgb and part 2_pre_def_rgb. Post-defoliation frames are stored in 205_Post_Def_rgb, Post_def_rgb_part1, part3_post_def_rgb, and part4_post_def_rgb. Folder names are used only to organize the current dataset; the final paper should report acquisition dates, UAV platform, camera, flight altitude, overlap, weather, and field layout.")
    paragraph(doc, "The current image stream contains dense cotton rows in which a single frame may contain several thousand bright boll-like structures. Manual instance annotation at this scale is expensive, and dense expert masks can take days per high-resolution image. This motivates a high-confidence candidate strategy: raw detections are retained for count and audit, while measurement-ready subsets are filtered by lint fraction, green canopy penalty, size consistency, visibility, brightness, and depth evidence.")
    paragraph(doc, "The present pipeline uses a user-specified image scale for proxy measurements. For final metric reporting, this scale must be replaced or validated with ground sampling distance, camera calibration, ground control points, RTK/GPS metadata, orthomosaic geometry, or direct physical boll measurements. Until that step is complete, length, width, diameter, and volume should be called proxy traits.")
    table(doc, "Table 2 Dataset fields required before final submission.", ["Field", "Current status", "Why it matters"], [
        ["Pre/post phase", "Available from folder names", "Defines visibility intervention"],
        ["Frame count", "To be audited", "Controls split and leakage reporting"],
        ["Camera and UAV model", "To be inserted", "Required by ICPA equipment reporting"],
        ["GSD or scale", "Proxy in MVP", "Needed for mm and mm3 measurements"],
        ["GCP/pose/orthomosaic", "To be verified", "Needed for metric plot mapping"],
        ["Manual validation labels", "To be created", "Needed for count, mask, and trait error"],
    ], [1.25, 1.45, 3.3])

    doc.add_heading("4 Method", level=1)
    doc.add_heading("4.1 Overview", level=2)
    paragraph(doc, "The proposed pipeline proceeds from a UAV frame to a structured phenotyping record. First, the image phase is taken from the dataset folder or inferred from canopy greenness. Second, candidate bolls are detected using a phase-aware computer vision detector. Third, each candidate is refined with a SAM-style lint mask. Fourth, a morphology-aware depth proxy or a calibrated reconstruction module maps image evidence into a 3D review space. Fifth, mask shape and projected 3D points are used to compute proxy traits. Finally, measurement-ready candidates are summarized at image and plot-cell levels.")
    doc.add_heading("4.2 Cotton boll candidate detection", level=2)
    paragraph(doc, "The detector is intentionally simple and reproducible. The image is resized to a stable working scale, converted through contrast-limited adaptive histogram equalization, and processed with small and large top-hat filters to emphasize bright cotton lint. Otsu thresholding produces initial components. Contours are filtered by area, aspect ratio, saturation, value, and luminance. A greenness heuristic distinguishes pre- and post-defoliation when phase labels are not supplied. The detector returns raw candidate boxes, an adjusted count, and an annotated image for audit.")
    doc.add_heading("4.3 SAM-style boll mask extraction", level=2)
    paragraph(doc, "Each detector box is treated as a prompt. The local crop is padded, converted to HSV color space, and filtered for low-saturation, high-value lint pixels while suppressing green canopy. Morphological opening and closing remove isolated noise. The largest connected lint component is retained as the candidate mask. This stage produces mask area, oriented mask length, oriented mask width, and inputs to the extraction-confidence score. Official SAM or SAM 2 inference can replace this deterministic mask stage in the same interface, but only after mask quality is evaluated on expert-labeled cotton examples.")
    doc.add_heading("4.4 Proxy 3D reconstruction and mask-to-3D review", level=2)
    paragraph(doc, "The MVP app estimates a morphology-aware depth proxy from row position, brightness, lint likelihood, canopy greenness, and local texture. The depth image is converted into a colored point cloud for interactive review. Separately, pixels belonging to measurement-ready boll masks are projected into the same coordinate frame and exported as a boll-mask point cloud. The resulting viewer resembles a segmentation-to-3D review loop: the original image shows the mask evidence, while the 3D view highlights the selected boll evidence inside the reconstructed scene.")
    paragraph(doc, "This representation is useful for quality control and method development, but it is not a substitute for calibrated 3D reconstruction. A metric version should use camera poses, SfM/MVS, Gaussian Splatting with validated scale, RGB-D data, or ground-control-based orthomosaic geometry. The manuscript should report the current 3D output as a proxy review space unless such calibration is completed.")
    doc.add_heading("4.5 Trait estimation", level=2)
    paragraph(doc, "Let s denote the image scale in mm per pixel, l_px and w_px denote the major and minor axes of the extracted mask, and w_box and h_box denote the detector-box width and height. The pipeline reports mask length L, mask width W, coarse detector diameter D, ellipsoid volume proxy V, visibility v, and extraction confidence q. Equations (1)-(6) define the planned trait computations.")
    equations(doc)
    paragraph(doc, "The ellipsoid volume proxy assumes that a boll can be approximated by one major axis and two equal minor axes. This is a pragmatic approximation for ranking and comparison, not a physical volume measurement. When physical measurements become available, the model should report mean absolute error, relative volume error, and correlation against measured boll dimensions.")
    doc.add_heading("4.6 Plot-level mapping", level=2)
    paragraph(doc, "The current app overlays a 4 by 43 plot-grid proxy over the central study area and assigns measurement-ready candidates to cells using image-coordinate centers. Each cell stores boll count, mean diameter proxy, mean volume proxy, and mean extraction quality. This map supports rapid inspection of spatial density and failure regions. It should be described as an image-coordinate plot proxy until plot boundaries, orthomosaic coordinates, or camera poses allow metric field mapping.")
    doc.add_heading("4.7 Agronomist-in-the-loop reporting", level=2)
    paragraph(doc, "The optional decision layer receives measured morphology records and converts them into structured agronomic summaries. It should not be described as performing detection, segmentation, or reconstruction. If included, open-source language models should be evaluated for schema validity, expert agreement, hallucination rate, latency, and consistency. This layer is downstream of geometry and should remain separate from the core reconstruction claims.")

    doc.add_heading("5 Algorithms", level=1)
    algorithm(doc, "Algorithm 1. Mask-guided cotton boll phenotyping pipeline.", "UAV image I, phase p, scale s or calibration metadata, detector parameters, mask parameters.", "Measurement table, mask overlay, scene point cloud, boll-mask point cloud, and plot-cell summary.", [
        "Resolve the image phase from metadata or greenness-based phase inference.",
        "Detect raw cotton boll candidates with the phase-aware detector.",
        "For each candidate, extract a prompt-style lint mask and compute mask statistics.",
        "Score extraction quality using lint fraction, green penalty, visibility, size prior, brightness, and depth evidence.",
        "Keep a measurement-ready subset for trait estimation and quality-control visualization.",
        "Project selected mask pixels into the 3D review coordinate system.",
        "Export image overlays, CSV trait records, full-scene PLY, boll-mask PLY, and plot-cell summaries.",
    ])
    algorithm(doc, "Algorithm 2. SAM-style boll mask extraction.", "Candidate crop C and detector box b.", "Binary lint mask M and mask-shape statistics.", [
        "Pad the detector box to include local context around the candidate.",
        "Convert the crop to HSV color space and compute excess-green suppression.",
        "Select low-saturation, high-value pixels consistent with cotton lint.",
        "Apply morphological opening and closing to remove isolated noise and fill small gaps.",
        "Compute connected components and retain the largest plausible lint component.",
        "Estimate mask area, oriented length, and oriented width from the retained component.",
    ])
    algorithm(doc, "Algorithm 3. Mask-to-3D projection and trait estimation.", "Mask M, depth or calibrated geometry Z, scale s, and candidate metadata.", "Projected boll-mask points and proxy trait vector.", [
        "Collect all foreground pixels from the mask and subsample if necessary for interactive rendering.",
        "Map original image coordinates to the depth or reconstruction grid.",
        "Back-project each selected pixel into the review coordinate system.",
        "Assign candidate-specific colors for visualization and export a boll-mask PLY.",
        "Compute L, W, D, V, v, and q using the mask, detector box, scale, and confidence terms.",
        "Store the resulting trait vector with candidate identity and phase metadata.",
    ])
    algorithm(doc, "Algorithm 4. Plot-cell aggregation.", "Measurement-ready candidates R and plot grid G.", "Cell-level count, trait means, and confidence summaries.", [
        "Define the row-column grid in image coordinates or calibrated field coordinates.",
        "For each candidate, compute the center point from the detector box or mask centroid.",
        "Assign the candidate to the corresponding grid cell when the center lies inside the study region.",
        "For each occupied cell, compute count, mean trait values, mean confidence, and coverage.",
        "Rank cells by count, uncertainty, or pre/post change for inspection and reporting.",
    ])

    doc.add_heading("References", level=1)
    for ref in REFERENCES:
        paragraph(doc, ref)

    doc.save(DOCX_OUT)
    MD_OUT.write_text(
        f"# {TITLE}\n\nThis Word-first draft has been completed through Algorithm 4. "
        "It removes running page furniture and internal-only notes, uses author-year citations, "
        "and preserves the ICPA accuracy boundary that all uncalibrated dimensions are proxy measurements.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_doc()
    print(f"Wrote {DOCX_OUT}")
    print(f"Wrote {MD_OUT}")

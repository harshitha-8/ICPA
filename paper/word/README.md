# Word Manuscript Workflow

ICPA requires Microsoft Word for the final manuscript upload. The working submission manuscript should be developed in this folder.

## Official Constraint

The 2026 ICPA and ConBAP full-paper instruction PDF states that Microsoft Word is the only accepted file format and that authors must use the MS Word template. The PDF also gives **22 May 2026** as the manuscript deadline. The public page exposes the template download button, but the direct template file was not accessible through a stable public URL during setup. Until the official `.docx` template is obtained through the browser or abstract-management portal, `icpa_manuscript_draft.docx` provides a Word-first working manuscript with ICPA-compatible structure.

## Files

- `build_icpa_docx.py`: reproducible builder for the working Word manuscript.
- `icpa_manuscript_draft.docx`: generated Word-first skeleton manuscript.
- `icpa_2026_mask_guided_cotton_until_algorithms.docx`: current Word-first
  manuscript draft completed through Algorithm 4.
- `icpa_2026_mask_guided_cotton_until_algorithms.md`: short Markdown source
  note for the current manuscript draft.
- `_qa/`: rendered page images for layout QA; not intended for submission.

Note: the current machine does not expose LibreOffice/`soffice`, so the new
DOCX skeleton has been structurally checked but not visually rendered to PNG.
Run the render QA step once LibreOffice is available.

## Recommended Practice

Write and revise in `icpa_manuscript_draft.docx`, then transfer styles into the official ICPA Word template once it is downloaded. Keep all figures and tables embedded inline, because ICPA requires a camera-ready Word manuscript.

Use the PDF constraints while drafting:

- Maximum 15 pages including figures, tables, graphs, and references.
- Single-spaced, camera-ready manuscript.
- No more than three displayed heading levels.
- Objective/passive technical prose; avoid personal pronouns and royal "we".
- SI units, especially `mm` and `m`.
- Author-year citations and alphabetized reference list.

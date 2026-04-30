# Word Manuscript Workflow

ICPA requires Microsoft Word for the final manuscript upload. The LaTeX files in `paper/latex/` are retained only as a drafting/reference source; the working submission manuscript should be developed in this folder.

## Official Constraint

The ICPA author-instruction page states that Microsoft Word is the only accepted file format and that authors must use the MS Word template. The public page exposes the template download button, but the direct template file was not accessible through a stable public URL during setup. Until the official `.docx` template is obtained through the browser or abstract-management portal, `icpa_manuscript_draft.docx` provides a Word-first working manuscript with ICPA-compatible structure.

## Files

- `build_icpa_docx.py`: reproducible builder for the working Word manuscript.
- `icpa_manuscript_draft.docx`: generated Word-first skeleton manuscript.
- `_qa/`: rendered page images for layout QA; not intended for submission.

## Recommended Practice

Write and revise in `icpa_manuscript_draft.docx`, then transfer styles into the official ICPA Word template once it is downloaded. Keep all figures and tables embedded inline, because ICPA requires a camera-ready Word manuscript.

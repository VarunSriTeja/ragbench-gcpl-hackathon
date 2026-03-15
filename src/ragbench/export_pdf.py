from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]


def _read_markdown_lines(path: Path) -> list[str]:
    return [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]


def _strip_markdown(text: str) -> str:
    cleaned = text.replace("`", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("#", "")
    return cleaned.strip()


def export_submission_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=0,
    )

    story = [
        Paragraph("GCPL Hackathon Submission Summary — Option B", title_style),
        Spacer(1, 0.4 * cm),
        Paragraph("RAG System with Retrieval Benchmarking", heading_style),
        Paragraph(
            "This PDF consolidates the architecture, evaluation approach, results, and demo guidance for the submission package.",
            body_style,
        ),
    ]

    deck_path = ROOT / "docs" / "submission_deck.md"
    for raw_line in _read_markdown_lines(deck_path):
        line = _strip_markdown(raw_line)
        if not line:
            continue
        if raw_line.startswith("## "):
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(line, heading_style))
        elif raw_line.startswith("- "):
            story.append(Paragraph(line[2:], bullet_style, bulletText="•"))
        else:
            story.append(Paragraph(line, body_style))

    summary_path = ROOT / "artifacts" / "benchmark_summary.md"
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Latest benchmark summary", heading_style))

    lines = _read_markdown_lines(summary_path)
    table_rows: list[list[str]] = []
    in_table = False
    for line in lines:
        if line.startswith("| chunking "):
            in_table = True
            table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        if in_table and not line.startswith("|"):
            break

    if table_rows:
        table = Table(table_rows, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF9")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEADING", (0, 0), (-1, -1), 10),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ]
            )
        )
        story.append(table)

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Demo assets", heading_style))
    story.append(Paragraph("- Architecture notes: docs/architecture.md", bullet_style, bulletText="•"))
    story.append(Paragraph("- Demo walkthrough: docs/demo_script.md", bullet_style, bulletText="•"))
    story.append(Paragraph("- Benchmark JSON: artifacts/eval_results_semantic.json", bullet_style, bulletText="•"))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    doc.build(story)

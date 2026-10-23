"""Export the Review 3 Markdown report as a formatted Word document.

Uses only Python's standard library so the document can be regenerated without
installing a third-party package.
"""

from __future__ import annotations

import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "REVIEW_3_PREPARATION.md"
OUTPUT = ROOT / "SysSense_AI_Project_Review_3_Updated.docx"

W_URI = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_URI = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_URI = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_URI = "http://schemas.openxmlformats.org/package/2006/content-types"
DC_URI = "http://purl.org/dc/elements/1.1/"
DCTERMS_URI = "http://purl.org/dc/terms/"
CP_URI = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
XSI_URI = "http://www.w3.org/2001/XMLSchema-instance"
XML_URI = "http://www.w3.org/XML/1998/namespace"

W = f"{{{W_URI}}}"
R = f"{{{R_URI}}}"
PKG = f"{{{PKG_URI}}}"
CT = f"{{{CT_URI}}}"

ET.register_namespace("w", W_URI)
ET.register_namespace("r", R_URI)
ET.register_namespace("rel", PKG_URI)
ET.register_namespace("ct", CT_URI)
ET.register_namespace("dc", DC_URI)
ET.register_namespace("dcterms", DCTERMS_URI)
ET.register_namespace("cp", CP_URI)
ET.register_namespace("xsi", XSI_URI)


def node(parent: ET.Element, tag: str, attrs: dict[str, str] | None = None) -> ET.Element:
    element = ET.SubElement(parent, W + tag)
    for key, value in (attrs or {}).items():
        element.set(W + key, value)
    return element


def set_text(element: ET.Element, value: str) -> None:
    element.text = value
    if value[:1].isspace() or value[-1:].isspace():
        element.set(f"{{{XML_URI}}}space", "preserve")


def add_run(parent: ET.Element, value: str, *, bold: bool = False,
            italic: bool = False, code: bool = False, color: str | None = None,
            size: int | None = None) -> None:
    run = node(parent, "r")
    props = node(run, "rPr")
    fonts = node(props, "rFonts", {"ascii": "Consolas" if code else "Calibri",
                                    "hAnsi": "Consolas" if code else "Calibri"})
    if bold:
        node(props, "b")
    if italic:
        node(props, "i")
    if code:
        node(props, "color", {"val": "334155"})
    if color:
        node(props, "color", {"val": color})
    if size:
        node(props, "sz", {"val": str(size)})
    text = node(run, "t")
    set_text(text, value)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*|\[[^\]]+\]\([^)]+\))")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


class WordReport:
    def __init__(self) -> None:
        self.document = ET.Element(W + "document")
        self.body = node(self.document, "body")
        self.relationships: list[tuple[str, str, str]] = []
        self.next_relationship = 3

    def add_inline(self, paragraph: ET.Element, text: str, *, default_bold: bool = False,
                   default_color: str | None = None, default_size: int | None = None) -> None:
        parts = INLINE.split(text)
        for part in parts:
            if not part:
                continue
            if part.startswith("[") and "](" in part:
                match = LINK.fullmatch(part)
                if match:
                    rel_id = f"rId{self.next_relationship}"
                    self.next_relationship += 1
                    self.relationships.append((rel_id, match.group(2), "External"))
                    hyperlink = ET.SubElement(paragraph, W + "hyperlink", {R + "id": rel_id})
                    add_run(hyperlink, match.group(1), color="0563C1")
                    continue
            if part.startswith("**") and part.endswith("**"):
                add_run(paragraph, part[2:-2], bold=True, color=default_color,
                        size=default_size)
            elif part.startswith("`") and part.endswith("`"):
                add_run(paragraph, part[1:-1], code=True, size=19)
            elif part.startswith("*") and part.endswith("*"):
                add_run(paragraph, part[1:-1], italic=True, color=default_color,
                        size=default_size)
            else:
                add_run(paragraph, part, bold=default_bold, color=default_color,
                        size=default_size)

    def add_paragraph(self, text: str = "", *, style: str | None = None,
                      alignment: str | None = None, code: bool = False,
                      indent: bool = False) -> ET.Element:
        paragraph = node(self.body, "p")
        ppr = node(paragraph, "pPr")
        if style:
            node(ppr, "pStyle", {"val": style})
        if alignment:
            node(ppr, "jc", {"val": alignment})
        if indent:
            node(ppr, "ind", {"left": "420", "hanging": "180"})
        if code:
            node(ppr, "shd", {"fill": "F1F5F9", "val": "clear"})
            node(ppr, "ind", {"left": "140", "right": "140"})
            add_run(paragraph, text, code=True, size=18)
        else:
            self.add_inline(paragraph, text)
        return paragraph

    def add_table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        table = node(self.body, "tbl")
        props = node(table, "tblPr")
        node(props, "tblW", {"w": "10800", "type": "dxa"})
        node(props, "tblLayout", {"type": "autofit"})
        borders = node(props, "tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            node(borders, edge, {"val": "single", "sz": "4", "space": "0",
                                 "color": "CBD5E1"})
        margins = node(props, "tblCellMar")
        for edge, width in (("top", "70"), ("left", "90"),
                            ("bottom", "70"), ("right", "90")):
            node(margins, edge, {"w": width, "type": "dxa"})

        column_count = max(len(row) for row in rows)
        grid = node(table, "tblGrid")
        for _ in range(column_count):
            ET.SubElement(grid, W + "gridCol", {W + "w": str(10800 // column_count)})

        for row_index, row_data in enumerate(rows):
            row = node(table, "tr")
            if row_index == 0:
                node(node(row, "trPr"), "tblHeader", {"val": "true"})
            for column_index in range(column_count):
                value = row_data[column_index] if column_index < len(row_data) else ""
                cell = node(row, "tc")
                tcpr = node(cell, "tcPr")
                node(tcpr, "tcW", {"w": str(10800 // column_count), "type": "dxa"})
                if row_index == 0:
                    node(tcpr, "shd", {"fill": "17365D", "val": "clear"})
                elif row_index % 2 == 0:
                    node(tcpr, "shd", {"fill": "F3F6FA", "val": "clear"})
                paragraph = node(cell, "p")
                ppr = node(paragraph, "pPr")
                node(ppr, "spacing", {"after": "20", "line": "230", "lineRule": "auto"})
                self.add_inline(paragraph, value,
                                default_bold=(row_index == 0),
                                default_color=("FFFFFF" if row_index == 0 else None),
                                default_size=18)

    def add_architecture(self) -> None:
        self.add_paragraph(
            "Operating system  →  psutil collector  →  SQLite  →  FastAPI  →  React dashboard",
            style="Architecture", alignment="center")
        self.add_paragraph(
            "FastAPI serves Random Forest/LSTM forecasts and Isolation Forest/rule-based alerts.",
            style="Caption", alignment="center")

    @staticmethod
    def is_table_separator(line: str) -> bool:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)

    def parse_markdown(self, source: str) -> None:
        lines = source.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            if line.startswith("```"):
                code_lines: list[str] = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                i += 1
                if code_lines and code_lines[0].strip().startswith("flowchart"):
                    self.add_architecture()
                else:
                    for code_line in code_lines:
                        self.add_paragraph(code_line or " ", code=True)
                continue

            if line.startswith("|"):
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i].strip())
                    i += 1
                table_rows = []
                for table_line in table_lines:
                    if self.is_table_separator(table_line):
                        continue
                    table_rows.append([cell.strip() for cell in
                                       table_line.strip("|").split("|")])
                self.add_table(table_rows)
                continue

            heading = re.match(r"^(#{1,3})\s+(.*)$", line)
            if heading:
                level = len(heading.group(1))
                style = "Title" if level == 1 else f"Heading{level - 1}"
                self.add_paragraph(heading.group(2), style=style,
                                   alignment="center" if level == 1 else None)
                i += 1
                continue

            numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
            if numbered:
                self.add_paragraph(f"{numbered.group(1)}.  {numbered.group(2)}", indent=True)
                i += 1
                continue

            bullet = re.match(r"^[-*]\s+(.*)$", line)
            if bullet:
                self.add_paragraph(f"•  {bullet.group(1)}", indent=True)
                i += 1
                continue

            paragraph_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if (not next_line or next_line.startswith(("#", "|", "```")) or
                        re.match(r"^(\d+)\.\s+", next_line) or
                        re.match(r"^[-*]\s+", next_line)):
                    break
                paragraph_lines.append(next_line)
                i += 1
            self.add_paragraph(" ".join(paragraph_lines))


def styles_xml() -> ET.Element:
    styles = ET.Element(W + "styles")
    defaults = node(styles, "docDefaults")
    rpr_default = node(node(defaults, "rPrDefault"), "rPr")
    node(rpr_default, "rFonts", {"ascii": "Calibri", "hAnsi": "Calibri"})
    node(rpr_default, "sz", {"val": "20"})
    ppr_default = node(node(defaults, "pPrDefault"), "pPr")
    node(ppr_default, "spacing", {"after": "100", "line": "276", "lineRule": "auto"})

    definitions = [
        ("Normal", "paragraph", None, None, None),
        ("Title", "paragraph", "2B579A", "48", "center"),
        ("Heading1", "paragraph", "17365D", "32", None),
        ("Heading2", "paragraph", "1F4E79", "26", None),
        ("Heading3", "paragraph", "365F91", "23", None),
        ("Architecture", "paragraph", "17365D", "22", "center"),
        ("Caption", "paragraph", "475569", "19", "center"),
    ]
    for style_id, style_type, color, size, alignment in definitions:
        style = ET.SubElement(styles, W + "style", {W + "type": style_type,
                                                       W + "styleId": style_id})
        if style_id == "Normal":
            style.set(W + "default", "1")
        node(style, "name", {"val": style_id})
        if style_id != "Normal":
            node(style, "basedOn", {"val": "Normal"})
            node(style, "next", {"val": "Normal"})
        ppr = node(style, "pPr")
        if alignment:
            node(ppr, "jc", {"val": alignment})
        node(ppr, "spacing", {"before": "200" if style_id.startswith("Heading") else "80",
                               "after": "100"})
        if style_id.startswith("Heading"):
            node(ppr, "keepNext")
        rpr = node(style, "rPr")
        if color:
            node(rpr, "color", {"val": color})
        if size:
            node(rpr, "sz", {"val": size})
        if style_id != "Caption":
            node(rpr, "b")
    return styles


def build_package(report: WordReport) -> None:
    sect = node(report.body, "sectPr")
    footer_ref = node(sect, "footerReference", {"type": "default"})
    footer_ref.set(R + "id", "rId2")
    node(sect, "pgSz", {"w": "12240", "h": "15840"})
    node(sect, "pgMar", {"top": "900", "right": "900", "bottom": "900",
                         "left": "900", "header": "450", "footer": "450",
                         "gutter": "0"})

    document_rels = ET.Element(PKG + "Relationships")
    for rel_id, target, rel_type in [
        ("rId1", "styles.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"),
        ("rId2", "footer1.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer"),
    ]:
        ET.SubElement(document_rels, PKG + "Relationship", {"Id": rel_id,
            "Type": rel_type, "Target": target})
    for rel_id, target, _ in report.relationships:
        ET.SubElement(document_rels, PKG + "Relationship", {"Id": rel_id,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
            "Target": target, "TargetMode": "External"})

    root_rels = ET.Element(PKG + "Relationships")
    ET.SubElement(root_rels, PKG + "Relationship", {"Id": "rId1",
        "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
        "Target": "word/document.xml"})
    ET.SubElement(root_rels, PKG + "Relationship", {"Id": "rId2",
        "Type": "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties",
        "Target": "docProps/core.xml"})
    ET.SubElement(root_rels, PKG + "Relationship", {"Id": "rId3",
        "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties",
        "Target": "docProps/app.xml"})

    content_types = ET.Element(CT + "Types")
    ET.SubElement(content_types, CT + "Default", {"Extension": "rels",
        "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(content_types, CT + "Default", {"Extension": "xml",
        "ContentType": "application/xml"})
    for part, content_type in [
        ("/word/document.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"),
        ("/word/styles.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"),
        ("/word/footer1.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"),
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml"),
    ]:
        ET.SubElement(content_types, CT + "Override", {"PartName": part,
            "ContentType": content_type})

    footer = ET.Element(W + "ftr")
    footer_p = node(footer, "p")
    footer_ppr = node(footer_p, "pPr")
    node(footer_ppr, "jc", {"val": "center"})
    add_run(footer_p, "SysSense AI  •  Project Review 3  |  Page ", color="64748B", size=18)
    field = node(footer_p, "fldSimple", {"instr": " PAGE "})
    field_run = node(field, "r")
    node(field_run, "t").text = "1"

    core = ET.Element(f"{{{CP_URI}}}coreProperties")
    ET.SubElement(core, f"{{{DC_URI}}}title").text = "SysSense AI — Project Review 3 Report"
    ET.SubElement(core, f"{{{DC_URI}}}creator").text = "Rohan Kumar; Shreyans Bohra"
    ET.SubElement(core, f"{{{DC_URI}}}subject").text = "Operating Systems Lab Project Review 3"
    ET.SubElement(core, f"{{{DC_URI}}}description").text = "Review 3 report prepared from repository evidence and the supplied rubric."
    ET.SubElement(core, f"{{{DCTERMS_URI}}}created", {f"{{{XSI_URI}}}type": "dcterms:W3CDTF"}).text = "2026-10-18T00:00:00Z"

    app = ET.Element("{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Properties")
    ET.SubElement(app, "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Application").text = "SysSense AI Review 3 report exporter"

    parts = {
        "[Content_Types].xml": content_types,
        "_rels/.rels": root_rels,
        "word/document.xml": report.document,
        "word/_rels/document.xml.rels": document_rels,
        "word/styles.xml": styles_xml(),
        "word/footer1.xml": footer,
        "docProps/core.xml": core,
        "docProps/app.xml": app,
    }
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for part_name, element in parts.items():
            package.writestr(part_name, ET.tostring(element, encoding="utf-8",
                                                     xml_declaration=True))


def main() -> None:
    report = WordReport()
    report.parse_markdown(SOURCE.read_text(encoding="utf-8"))
    build_package(report)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()

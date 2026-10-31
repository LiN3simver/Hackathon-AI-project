"""Full assistant results as an .xlsx file; the chat shows at most CARD_ROWS photo groups.

Written with the standard library (an .xlsx is a zip of XML parts), so no
spreadsheet dependency is needed for a few thousand plain rows.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from decimal import Decimal
from xml.sax.saxutils import escape

from app.schemas.history import EvidenceRow, HistoryQueryRequest
from app.services.history_evidence import RESPONSIBILITY_LABELS, build_evidence

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _column(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell(ref: str, value, style: int = 0) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        value = "是" if value else "否"
    if isinstance(value, int | float | Decimal):
        return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    text = escape(_ILLEGAL.sub("", str(value)))
    return f'<c r="{ref}" t="inlineStr" s="{style}"><is><t xml:space="preserve">{text}</t></is></c>'


def _sheet(headers: list[str], rows: list[list], widths: list[int]) -> str:
    body = ["<row r=\"1\">" + "".join(_cell(f"{_column(i)}1", h, 1) for i, h in enumerate(headers)) + "</row>"]
    for number, row in enumerate(rows, start=2):
        body.append(f'<row r="{number}">' + "".join(
            _cell(f"{_column(i)}{number}", value) for i, value in enumerate(row)) + "</row>")
    cols = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{width}" customWidth="1"/>'
                   for i, width in enumerate(widths))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/>'
            f'</sheetView></sheetViews><cols>{cols}</cols><sheetData>{"".join(body)}</sheetData></worksheet>')


def build_xlsx(sheets: list[tuple[str, list[str], list[list], list[int]]]) -> bytes:
    """sheets: (name, headers, rows, column widths)."""
    output = io.BytesIO()
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                      'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                      for i in range(1, len(sheets) + 1)) + "</Types>"))
        archive.writestr("_rels/.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>'))
        archive.writestr("xl/workbook.xml", (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="{main}" xmlns:r="{rel}"><sheets>'
            + "".join(f'<sheet name="{escape(name[:31])}" sheetId="{i}" r:id="rId{i}"/>'
                      for i, (name, *_rest) in enumerate(sheets, start=1)) + "</sheets></workbook>"))
        archive.writestr("xl/_rels/workbook.xml.rels", (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>'
                      for i in range(1, len(sheets) + 1))
            + f'<Relationship Id="rId{len(sheets) + 1}" Type="{rel}/styles" Target="styles.xml"/></Relationships>'))
        archive.writestr("xl/styles.xml", (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="{main}">'
            '<fonts count="2"><font><sz val="11"/><name val="Microsoft JhengHei"/></font>'
            '<font><b/><sz val="11"/><name val="Microsoft JhengHei"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
            '<cellXfs count="2"><xf fontId="0"/><xf fontId="1" applyFont="1"/></cellXfs></styleSheet>'))
        for i, (_name, headers, rows, widths) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{i}.xml", _sheet(headers, rows, widths))
    return output.getvalue()


def _presence(value: bool | None) -> str:
    return "是" if value else "否" if value is False else "未記錄"


def _photo_link(row: EvidenceRow, api_base: str, slot: str) -> str:
    if slot not in row.photo_slots:
        return ""
    if row.photo_source == "rag_seed" and row.event_id:
        return f"{api_base}/admin/rag-seeds/{row.event_id}/images/{slot}"
    if row.photo_source == "damage_case" and row.damage_case_id:
        return f"{api_base}/damage/cases/{row.damage_case_id}/images/{slot}"
    return ""


def export_history(request: HistoryQueryRequest, store, api_base: str) -> tuple[str, bytes]:
    """Every matching record of the question (no 30-row cap), plus per-person and risk sheets."""
    from app.services.history_answer import _today
    from app.services.history_query import query_history_data

    response = query_history_data(request, store=store)
    card = build_evidence(request, response, store, today=_today(), row_limit=None).card
    headers = ["編號", "日期", "還車日期", "承租人", "車牌", "類型", "責任", "計入統計", "借車照片有此損傷",
               "還車照片有此損傷", "應賠", "已賠", "標題", "描述", "借車照片連結（需登入）", "還車照片連結（需登入）"]
    rows = [[row.ref, str(row.occurred_on or ""), str(row.return_date or ""), row.renter_name, row.license_plate,
             row.kind_label, RESPONSIBILITY_LABELS[row.responsibility], row.counted,
             _presence(row.present_at_pickup) if row.kind == "damage" else "",
             _presence(row.present_at_return) if row.kind == "damage" else "",
             Decimal(row.compensation_amount) if row.compensation_amount else None,
             Decimal(row.paid_amount) if row.paid_amount else None, row.title, row.summary,
             _photo_link(row, api_base, "pickup"), _photo_link(row, api_base, "return")] for row in card.rows]
    sheets = [("依據紀錄", headers, rows, [6, 12, 12, 12, 12, 8, 12, 9, 14, 14, 12, 12, 24, 48, 60, 60])]
    if card.people:
        # Same columns as the on-screen table: only the topics the question asked about.
        damage, litter = "damage" in card.people_columns, "litter" in card.people_columns
        people_headers = ["使用者", "車牌"] + (["責任車損", "待確認", "爭議"] if damage else []) + (
            ["垃圾"] if litter else []) + (["應賠", "未賠"] if damage else [])
        people_rows = [[person.name, "、".join(person.license_plates)]
                       + ([person.caused_damage_count, person.unconfirmed_damage_count, person.disputed_count]
                          if damage else [])
                       + ([person.litter_count] if litter else [])
                       + ([Decimal(person.compensation_amount), Decimal(person.outstanding_amount)] if damage else [])
                       for person in card.people]
        sheets.append(("每人統計", people_headers, people_rows,
                       [14, 20] + [10, 8, 8] * damage + [8] * litter + [12, 12] * damage))
    if card.risk:
        risk_headers = ["排名", "使用者", "風險等級", "責任車損", "垃圾", "爭議", "未賠", "租借次數", "原因"]
        sheets.append(("風險排名", risk_headers, [
            [row.rank, row.name, row.level_label, row.caused_damage_count, row.litter_count, row.disputed_count,
             Decimal(row.outstanding_amount), row.rental_count, "；".join(row.reasons)] for row in card.risk],
            [6, 14, 16, 10, 8, 8, 12, 10, 50]))
    sheets.append(("查詢條件", ["項目", "內容"], [
        ["問題", request.question], ["對象", card.scope_label], ["期間", card.period_label],
        ["主題", card.topic_label], ["重點", card.headline], ["紀錄筆數", len(card.rows)],
        ["匯出時間", datetime.now().strftime("%Y-%m-%d %H:%M")]], [12, 80]))
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"借還車AI助理_{stamp}.xlsx", build_xlsx(sheets)

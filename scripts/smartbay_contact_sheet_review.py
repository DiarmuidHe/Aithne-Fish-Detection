"""Build an offline, exportable review UI from a SmartBay contact-sheet queue."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

import cv2

TILE_WIDTH = 400
TILE_HEIGHT = 252
SHEET_COLUMNS = 4
REVIEW_FIELDS = [
    "window_id",
    "camera",
    "day",
    "split",
    "evidence",
    "contact_sheet",
    "decision",
    "replacement_reason",
    "review_notes",
]
DEFAULT_NOTES = {
    "strong": "Provisional approve: plausible fish evidence in a reviewable scene; labels still require human verification.",
    "weak": "Provisional approve: useful ambiguous or hard-negative scene; labels still require human verification.",
    "none": "Provisional approve: usable negative/control scene; absence of fish still requires human verification.",
    "murky": "Provisional approve: difficult water conditions, but scene content remains discernible.",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"review queue is empty: {path}")
    missing = set(REVIEW_FIELDS) - set(rows[0])
    if missing:
        raise ValueError(f"review queue lacks columns: {sorted(missing)}")
    return rows


def apply_proposals(rows: list[dict[str, str]], path: Path | None) -> list[dict[str, str]]:
    proposals = {}
    if path is not None:
        proposals = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(proposals, dict):
            raise ValueError("proposals JSON must be an object keyed by window_id")
    known = {row["window_id"] for row in rows}
    unknown = set(proposals) - known
    if unknown:
        raise ValueError(f"proposals contain unknown window IDs: {sorted(unknown)}")
    result = []
    for original in rows:
        row = dict(original)
        proposal = proposals.get(row["window_id"], {})
        row["decision"] = proposal.get("decision", row["decision"] or "approve")
        row["replacement_reason"] = proposal.get("replacement_reason", row["replacement_reason"])
        row["review_notes"] = proposal.get(
            "review_notes", row["review_notes"] or DEFAULT_NOTES[row["evidence"]]
        )
        if row["decision"] not in {"approve", "reject"}:
            raise ValueError(f"invalid decision for {row['window_id']}: {row['decision']}")
        result.append(row)
    return result


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in REVIEW_FIELDS} for row in rows)


def extract_tiles(rows: list[dict[str, str]], queue_dir: Path, tile_dir: Path) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    sheet_positions: Counter[str] = Counter()
    sheets = {}
    for row in rows:
        sheet_name = row["contact_sheet"]
        position = sheet_positions[sheet_name]
        sheet_positions[sheet_name] += 1
        if sheet_name not in sheets:
            path = queue_dir / sheet_name
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"could not read contact sheet: {path}")
            sheets[sheet_name] = image
        sheet = sheets[sheet_name]
        tile_row, tile_column = divmod(position, SHEET_COLUMNS)
        top, left = tile_row * TILE_HEIGHT, tile_column * TILE_WIDTH
        tile = sheet[top : top + TILE_HEIGHT, left : left + TILE_WIDTH]
        if tile.shape[:2] != (TILE_HEIGHT, TILE_WIDTH):
            raise ValueError(
                f"contact sheet tile {position} is incomplete for {row['window_id']}: {sheet_name}"
            )
        target = tile_dir / f"{row['window_id']}.jpg"
        if not cv2.imwrite(str(target), tile, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"could not write review tile: {target}")


def _option(value: str, selected: bool) -> str:
    return f'<option value="{value}"{" selected" if selected else ""}>{value}</option>'


def write_html(rows: list[dict[str, str]], path: Path, tile_dir_name: str) -> None:
    counts = Counter(row["decision"] for row in rows)
    cards = []
    for index, row in enumerate(rows):
        escaped = {key: html.escape(value, quote=True) for key, value in row.items()}
        cards.append(
            f'''<article class="card" data-index="{index}" data-camera="{escaped["camera"]}"
 data-split="{escaped["split"]}" data-evidence="{escaped["evidence"]}" data-decision="{escaped["decision"]}">
<img src="{html.escape(tile_dir_name, quote=True)}/{escaped["window_id"]}.jpg" alt="{escaped["window_id"]}">
<div class="body"><h2>{escaped["window_id"]}</h2>
<p>{escaped["camera"]} · {escaped["split"]} · {escaped["evidence"]} · {escaped["day"]}</p>
<label>Decision <select class="decision">{_option("approve", row["decision"] == "approve")}{_option("reject", row["decision"] == "reject")}</select></label>
<label>Replacement reason <input class="reason" value="{escaped["replacement_reason"]}"></label>
<label>Review notes <textarea class="notes">{escaped["review_notes"]}</textarea></label></div></article>'''
        )
    data = json.dumps([{field: row[field] for field in REVIEW_FIELDS} for row in rows]).replace(
        "</", "<\\/"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartBay contact-sheet review</title>
<style>
:root{{--bg:#eef3f0;--panel:#fff;--ink:#14221b;--muted:#607067;--line:#cbd6d0;--ok:#186a3b;--bad:#a62c2c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#14221bf2;color:#fff;padding:14px 20px;box-shadow:0 2px 8px #0004}}
h1{{font-size:20px;margin:0 0 8px}}.controls{{display:flex;flex-wrap:wrap;gap:8px;align-items:center}}
button,select,input,textarea{{font:inherit}}button,.controls select{{padding:6px 10px;border:1px solid #829087;border-radius:6px}}
button{{cursor:pointer;background:#fff}}#summary{{margin-left:auto}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(410px,1fr));gap:14px;padding:16px}}
.card{{background:var(--panel);border:2px solid var(--line);border-radius:9px;overflow:hidden;box-shadow:0 2px 5px #0001}}
.card[data-decision="approve"]{{border-color:#78ae8b}}.card[data-decision="reject"]{{border-color:#d37d7d}}
.card img{{display:block;width:100%;max-width:400px;height:auto;margin:auto;background:#222}}
.body{{padding:10px}}h2{{font-size:15px;margin:0;overflow-wrap:anywhere}}p{{color:var(--muted);margin:3px 0 9px}}
label{{display:block;font-weight:600;margin-top:7px}}label select,label input,label textarea{{display:block;width:100%;margin-top:3px;padding:6px;border:1px solid var(--line);border-radius:5px;background:#fff}}
textarea{{min-height:64px;resize:vertical}}.hidden{{display:none}}.warning{{font-size:13px;color:#d9e4de}}
</style></head><body><header><h1>SmartBay window review</h1>
<div class="warning">Provisional visual decisions only. Confirm every card; screening boxes are not ground truth.</div>
<div class="controls"><select id="camera"><option value="">All cameras</option><option>SmartBayCam1</option><option>SmartBayCam2</option></select>
<select id="split"><option value="">All splits</option><option>train</option><option>validation</option><option>test</option><option>expert</option></select>
<select id="evidence"><option value="">All evidence</option><option>strong</option><option>weak</option><option>none</option><option>murky</option></select>
<select id="decision"><option value="">All decisions</option><option>approve</option><option>reject</option></select>
<button id="export">Export reviewed CSV</button><button id="reset">Reset provisional choices</button>
<span id="summary">{len(rows)} windows · {counts["approve"]} approve · {counts["reject"]} reject</span></div></header>
<main>{"".join(cards)}</main><script>
const initial={data}; const key='smartbay-review-'+location.pathname; let rows;
try{{rows=JSON.parse(localStorage.getItem(key))||structuredClone(initial)}}catch(e){{rows=structuredClone(initial)}}
const cards=[...document.querySelectorAll('.card')];
function syncCard(card){{const i=+card.dataset.index,r=rows[i];card.querySelector('.decision').value=r.decision;card.querySelector('.reason').value=r.replacement_reason;card.querySelector('.notes').value=r.review_notes;card.dataset.decision=r.decision}}
function save(){{localStorage.setItem(key,JSON.stringify(rows));const a=rows.filter(r=>r.decision==='approve').length;document.querySelector('#summary').textContent=`${{rows.length}} windows · ${{a}} approve · ${{rows.length-a}} reject`}}
cards.forEach(card=>{{syncCard(card);const i=+card.dataset.index;card.querySelector('.decision').onchange=e=>{{rows[i].decision=e.target.value;syncCard(card);save();filter()}};card.querySelector('.reason').oninput=e=>{{rows[i].replacement_reason=e.target.value;save()}};card.querySelector('.notes').oninput=e=>{{rows[i].review_notes=e.target.value;save()}}}});
function filter(){{for(const card of cards){{card.classList.toggle('hidden',!['camera','split','evidence','decision'].every(id=>!document.querySelector('#'+id).value||card.dataset[id]===document.querySelector('#'+id).value))}}}}
['camera','split','evidence','decision'].forEach(id=>document.querySelector('#'+id).onchange=filter);
function csvCell(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}}
document.querySelector('#export').onclick=()=>{{const fields={json.dumps(REVIEW_FIELDS)};const csv=[fields.join(','),...rows.map(r=>fields.map(f=>csvCell(r[f])).join(','))].join('\r\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='contact_sheet_review.csv';a.click();URL.revokeObjectURL(a.href)}};
document.querySelector('#reset').onclick=()=>{{if(confirm('Reset all edits to the provisional decisions?')){{rows=structuredClone(initial);cards.forEach(syncCard);save();filter()}}}};
save();filter();
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def build(queue: Path, output: Path, proposals: Path | None = None) -> dict[str, int]:
    rows = apply_proposals(read_rows(queue), proposals)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "contact_sheet_review_proposed.csv")
    extract_tiles(rows, queue.parent, output / "tiles")
    write_html(rows, output / "review.html", "tiles")
    return dict(Counter(row["decision"] for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.queue, args.output, args.proposals), indent=2))


if __name__ == "__main__":
    main()

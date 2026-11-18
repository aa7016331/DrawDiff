import os
import csv
import io
import base64
from pathlib import Path
from collections import Counter

import cv2
import fitz
import numpy as np
from flask import Flask, render_template_string, request, jsonify, Response


BASE = Path(__file__).resolve().parent
SYMBOL_DIR = BASE / "static" / "symbols"

ALLOWED = {"pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# 0～12 號符號固定分類名稱
SYMBOL_NAMES = {
    0: "住戶開關箱與弱電箱整合",
    1: "馬桶專用插座 H:30cm",
    2: "抽油煙機插座 H:天花板內",
    3: "雙聯插座 H:30cm／電視牆 H:80cm",
    4: "雙聯插座 H:110cm",
    5: "洗碗機專用插座 H:45cm",
    6: "微波烤箱專用插座 220V",
    7: "洗衣機用插座（防雨型）H:120cm",
    8: "廚房橫櫃燈用電源出線口 H:170cm",
    9: "暖風乾燥機電源出口 220V",
    10: "紅色緊急插座（弱電箱）",
    11: "紅色緊急插座（冰箱／電視牆）",
    12: "冷氣電源出口 220V",
}

# 分類參數：精準優先，無法清楚辨認時保留未分類。
CLASSIFY_CANVAS_SIZE = 176
CLASSIFY_EDGE_TOLERANCE = 2.5
CLASSIFY_SCALE_VARIANTS = (0.92, 1.0, 1.08)
CLASSIFY_CODE_MIN_SCORE = 0.70
CLASSIFY_CODE_MIN_MARGIN = 0.03
CLASSIFY_BODY_ONLY_MIN_SCORE = 0.86
CLASSIFY_BODY_ONLY_MIN_MARGIN = 0.10

# 範本內部的英文代碼位置（以各 PNG 原始像素座標表示）。
# 這些是符號圖形的一部分，使用影像特徵比對，不使用 OCR。
GLYPH_BOXES = {
    2: (40, 17, 47, 29),   # F
    4: (39, 14, 47, 27),   # H
    5: (39, 16, 47, 28),   # G
    6: (38, 17, 46, 28),   # K
    7: (38, 18, 48, 29),   # W
    9: (39, 15, 46, 28),   # H
    10: (40, 30, 48, 43),  # C
    11: (39, 38, 46, 47),  # E
    12: (39, 17, 46, 28),  # H
}
INTERNAL_GLYPH_BOXES = {
    8: (21, 8, 32, 25),    # J，位於圓形內部
}
CODE_SYMBOL_IDS = (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)
BODY_ONLY_SYMBOL_IDS = (1, 3)

# 代碼字形的跨類別驗證。H 由 4／9／12 共用；其他代碼各自獨立。
# 這一步不是 OCR，而是把紅圈候選中代碼所在固定區域的形狀，
# 與 13 張官方範本內的英文字形特徵逐一比對，防止把 H 誤當 W 等情況。
GLYPH_GROUPS = {
    "F": (2,),
    "H": (4, 9, 12),
    "G": (5,),
    "K": (6,),
    "W": (7,),
    "J": (8,),
    "C": (10,),
    "E": (11,),
}
SYMBOL_GLYPH_GROUP = {
    symbol_id: glyph_group
    for glyph_group, symbol_ids in GLYPH_GROUPS.items()
    for symbol_id in symbol_ids
}
# 若另一種字母的分數明顯高於候選所屬代碼，該候選不能通過。
# G/C 筆畫相近，採較寬容的衝突門檻，再由圓內結構與整體分數把關。
GLYPH_CODE_CONFLICT_MARGIN = {
    2: 0.035, 4: 0.010, 5: 0.070, 6: 0.030, 7: 0.025,
    8: 0.035, 9: 0.010, 10: 0.045, 11: 0.030, 12: 0.010,
}

# 範本中的圓心與半徑（原始範本像素座標）。
# 0 號為長方形箱體符號，並非圓形符號，不交由目前的圓形紅圈候選分類。
TEMPLATE_CIRCLE_GEOMETRY = {
    1: (26.0, 14.0, 13.0),
    2: (26.0, 14.0, 13.0),
    3: (26.0, 14.0, 13.0),
    4: (26.0, 14.0, 13.0),
    5: (26.0, 14.0, 13.0),
    6: (26.0, 14.0, 13.0),
    7: (26.0, 14.0, 13.0),
    8: (26.0, 14.0, 13.0),
    9: (26.0, 14.0, 13.0),
    10: (28.0, 28.0, 13.0),
    11: (26.0, 34.0, 13.0),
    12: (26.0, 14.0, 13.0),
}

PLAN_X0 = 0.11
PLAN_X1 = 0.73
PLAN_Y0 = 0.20
PLAN_Y1 = 0.84
RENDER_SCALE = 4.0

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024


INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DrawDiff｜13 種設備符號分類</title>
<style>
:root{font-family:Arial,"Noto Sans TC",sans-serif;color:#152432;background:#f2f5f7}
*{box-sizing:border-box}
body{margin:0;padding:24px}
main,header{max-width:1420px;margin:0 auto}
header{margin-bottom:20px}
h1{font-size:25px;margin:0 0 6px}
.sub{color:#627381;font-size:14px}
.card{background:#fff;border:1px solid #d7e0e6;border-radius:14px;padding:22px;margin-bottom:18px;box-shadow:0 2px 8px rgba(20,40,55,.04)}
h2{font-size:20px;margin:0 0 18px}
.upload-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.upload-box{border:1px dashed #aabac5;background:#fbfcfd;padding:16px;border-radius:10px}
.upload-box strong{display:block;margin-bottom:8px}
input[type=file]{display:block;width:100%;margin-top:8px}
.page-row{display:flex;align-items:center;gap:10px;margin-top:12px;color:#546675;font-size:14px}
input[type=number]{width:84px;padding:8px 9px;border:1px solid #cbd6dd;border-radius:8px;font-size:15px}
.actions{display:flex;gap:12px;margin-top:18px;flex-wrap:wrap}
button{border:0;border-radius:9px;padding:11px 18px;font-size:15px;cursor:pointer}
.primary{background:#164f71;color:white}
.secondary{background:#e9f0f4;color:#1b3a4e}
button:disabled{opacity:.5;cursor:not-allowed}
#status{margin-top:14px;min-height:20px;color:#5d6c77;font-size:14px;line-height:1.6}
.error{color:#b42318!important}
#result{display:none}
.summary{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.pill{background:#edf3f6;border-radius:999px;padding:10px 15px;font-weight:600}
.pill.plus{color:#13724e;background:#e8f5ee}
.pill.minus{color:#a33c32;background:#faeeec}
.main-result{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:10px 0 20px}
.metric{border:1px solid #e0e7eb;border-radius:10px;padding:14px}
.metric .label{font-size:13px;color:#687987}
.metric .value{font-size:28px;font-weight:700;margin-top:4px}
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.image-panel{border:1px solid #d7e0e6;background:#fafcfd;border-radius:10px;padding:12px;min-width:0}
.image-title{text-align:center;font-weight:700;margin-bottom:8px}
.image-panel img{width:100%;height:auto;display:block;background:#fff;border:1px solid #e5eaee}
.method{font-size:12px;color:#667784;margin-top:8px;line-height:1.45}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin-top:20px;min-width:720px}
th,td{border-bottom:1px solid #e3e9ed;padding:11px;text-align:center}
th{background:#edf2f5;position:sticky;top:0}
td:first-child,th:first-child{text-align:left}
tr.unknown{background:#fff8e8}
tr.total{font-weight:700;background:#edf2f5}
.note{color:#687987;font-size:13px;margin-top:12px;line-height:1.6}
@media(max-width:850px){body{padding:12px}.upload-grid,.compare-grid{grid-template-columns:1fr}.main-result{grid-template-columns:1fr 1fr}.card{padding:16px}}
</style>
</head>
<body>
<header>
  <h1>DrawDiff｜13 種設備符號分類</h1>
  <div class="sub">保留原本紅圈定位方式，以 13 種符號的內部圖形、英文代碼交叉驗證與四個正交方向進行分類。</div>
</header>
<main>
<section class="card">
  <h2>上傳圖面</h2>
  <form id="uploadForm">
    <div class="upload-grid">
      <div class="upload-box">
        <strong>原始圖</strong>
        <div class="sub">預設讀取 PDF 第 4 頁</div>
        <input id="oldFile" name="old" type="file" accept=".pdf,.jpg,.jpeg,.png,.bmp,.tif,.tiff" required>
        <div class="page-row">PDF 頁碼 <input id="oldPage" name="old_page" type="number" min="1" value="4"><span>（圖片檔會忽略）</span></div>
      </div>
      <div class="upload-box">
        <strong>設計師修改圖</strong>
        <div class="sub">預設讀取 PDF 第 3 頁</div>
        <input id="newFile" name="new" type="file" accept=".pdf,.jpg,.jpeg,.png,.bmp,.tif,.tiff" required>
        <div class="page-row">PDF 頁碼 <input id="newPage" name="new_page" type="number" min="1" value="3"><span>（圖片檔會忽略）</span></div>
      </div>
    </div>
    <div class="actions">
      <button class="primary" id="analyzeButton" type="submit">開始辨識與分類</button>
      <button class="secondary" id="exportButton" type="button" disabled>匯出增減表 CSV</button>
    </div>
  </form>
  <div id="status"></div>
</section>

<section class="card" id="result">
  <h2>13 種設備增減結果</h2>
  <div class="summary" id="summary"></div>
  <div class="main-result">
    <div class="metric"><div class="label">原圖紅圈偵測總數</div><div class="value" id="oldCount">—</div></div>
    <div class="metric"><div class="label">修改後紅圈偵測總數</div><div class="value" id="newCount">—</div></div>
    <div class="metric"><div class="label">紅圈總數淨增減</div><div class="value" id="netCount">—</div></div>
  </div>
  <div class="compare-grid">
    <div class="image-panel">
      <div class="image-title">原始圖（紅圈為原本偵測到的位置）</div>
      <img id="oldResult" alt="原始圖紅圈結果">
      <div class="method" id="oldMethod"></div>
    </div>
    <div class="image-panel">
      <div class="image-title">設計師修改圖（紅圈為原本偵測到的位置）</div>
      <img id="newResult" alt="修改圖紅圈結果">
      <div class="method" id="newMethod"></div>
    </div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>編號／設備項目</th><th>原圖數量</th><th>修改後數量</th><th>增加</th><th>減少</th><th>淨增減</th></tr></thead>
      <tbody id="resultRows"></tbody>
    </table>
  </div>
  <div class="note" id="resultNote"></div>
</section>
</main>
<script>
let latestRows=[];
const form=document.getElementById('uploadForm');
const statusEl=document.getElementById('status');
const analyzeButton=document.getElementById('analyzeButton');
const exportButton=document.getElementById('exportButton');

form.addEventListener('submit',async(event)=>{
  event.preventDefault();
  const oldFile=document.getElementById('oldFile').files[0];
  const newFile=document.getElementById('newFile').files[0];
  if(!oldFile||!newFile){statusEl.textContent='請同時選擇原始圖及修改後圖。';statusEl.className='error';return;}
  const payload=new FormData(form);
  analyzeButton.disabled=true;
  exportButton.disabled=true;
  statusEl.textContent='正在讀取圖面、定位紅圈並比對 13 種符號模板……';
  statusEl.className='';
  try{
    const response=await fetch('/analyze',{method:'POST',body:payload});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||'辨識失敗');

    document.getElementById('result').style.display='block';
    document.getElementById('oldCount').textContent=data.old_detected_count;
    document.getElementById('newCount').textContent=data.new_detected_count;
    const net=data.new_detected_count-data.old_detected_count;
    document.getElementById('netCount').textContent=(net>0?'+':'')+net;

    const summary=document.getElementById('summary');
    summary.replaceChildren();
    [
      ['原圖 '+data.old_detected_count+' 個',''],
      ['修改後 '+data.new_detected_count+' 個',''],
      ['增加 '+data.added_count+' 個','plus'],
      ['減少 '+data.removed_count+' 個','minus'],
      ['原圖未分類 '+data.old_unknown_count+' 個',''],
      ['修改後未分類 '+data.new_unknown_count+' 個',''],
      ['原圖代碼圖形確認 '+data.old_glyph_matched_count+' 個',''],
      ['修改圖代碼圖形確認 '+data.new_glyph_matched_count+' 個','']
    ].forEach(([label,cls])=>{
      const d=document.createElement('div');d.className='pill '+cls;d.textContent=label;summary.appendChild(d);
    });

    document.getElementById('oldResult').src=data.old_image;
    document.getElementById('newResult').src=data.new_image;
    document.getElementById('oldMethod').textContent=data.old_method+'。'+data.old_note;
    document.getElementById('newMethod').textContent=data.new_method+'。'+data.new_note;

    const body=document.getElementById('resultRows');
    body.replaceChildren();
    data.rows.forEach(row=>{
      const tr=document.createElement('tr');
      if(row.id==='unknown')tr.className='unknown';
      const name=document.createElement('td');
      name.textContent=row.id==='unknown'?'未分類（需人工核對）':String(row.id)+'｜'+row.name;
      tr.appendChild(name);
      [row.old_count,row.new_count,row.added_count,row.removed_count,(row.difference>0?'+':'')+row.difference].forEach(v=>{
        const td=document.createElement('td');td.textContent=v;tr.appendChild(td);
      });
      body.appendChild(tr);
    });

    const totalTr=document.createElement('tr');
    totalTr.className='total';
    const totalLabel=document.createElement('td');totalLabel.textContent='合計（含未分類）';totalTr.appendChild(totalLabel);
    [data.old_detected_count,data.new_detected_count,data.added_count,data.removed_count,(net>0?'+':'')+net].forEach(v=>{
      const td=document.createElement('td');td.textContent=v;totalTr.appendChild(td);
    });
    body.appendChild(totalTr);

    latestRows=data.rows;
    document.getElementById('resultNote').textContent=data.row_source;
    exportButton.disabled=false;
    statusEl.textContent='辨識與分類完成。請先核對左右圖面的紅圈位置，再檢查各設備分類及未分類數量。';
    statusEl.className='';
    document.getElementById('result').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(err){
    statusEl.textContent=err.message;
    statusEl.className='error';
  }finally{
    analyzeButton.disabled=false;
  }
});

exportButton.addEventListener('click',async()=>{
  if(!latestRows.length)return;
  try{
    const r=await fetch('/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rows:latestRows})});
    if(!r.ok)throw new Error('CSV 匯出失敗');
    const blob=await r.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download='DrawDiff_13種設備增減統計.csv';document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);
  }catch(err){statusEl.textContent=err.message;statusEl.className='error';}
});
</script>
</body>
</html>
"""


def _filename_ext(filename):
    return Path(filename or "").suffix.lower().lstrip(".")


def _jpeg_data_url(image_bgr, max_side=1900, quality=84):
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    ok, buffer = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("結果圖片編碼失敗")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _mark_centers(image_bgr, centers):
    """原樣保留：在已偵測到的設備中心畫紅圈。"""
    result = image_bgr.copy()
    h, w = result.shape[:2]

    for cx, cy, radius in centers:
        x = int(round(cx))
        y = int(round(cy))
        r = max(10, int(round(radius)))
        if not (0 <= x < w and 0 <= y < h):
            continue
        cv2.circle(
            result,
            (x, y),
            r,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return result


def _vector_socket_centers(page):
    """
    原樣保留：CAD 向量 PDF，以重複的四段 Bezier 圓形路徑定位。
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    candidates = []

    for drawing in drawings:
        items = drawing.get("items", [])
        if len(items) != 4 or not all(item[0] == "c" for item in items):
            continue

        rect = drawing.get("rect")
        if rect is None:
            continue

        visible_rect = rect * page.rotation_matrix
        width = float(visible_rect.width)
        height = float(visible_rect.height)

        if abs(width - height) > 0.25:
            continue

        diameter = (width + height) / 2.0
        if not (10.0 <= diameter <= 18.0):
            continue

        cx = (visible_rect.x0 + visible_rect.x1) / 2.0
        cy = (visible_rect.y0 + visible_rect.y1) / 2.0

        if not (PLAN_X0 * page_w <= cx <= PLAN_X1 * page_w):
            continue
        if not (PLAN_Y0 * page_h <= cy <= PLAN_Y1 * page_h):
            continue

        candidates.append((cx, cy, diameter))

    if len(candidates) < 8:
        return []

    size_counts = Counter(round(item[2], 1) for item in candidates)
    dominant_size, dominant_count = size_counts.most_common(1)[0]

    if dominant_count < 8:
        return []

    selected = [
        item for item in candidates
        if abs(item[2] - dominant_size) <= 0.25
    ]

    selected.sort(key=lambda item: item[2], reverse=True)
    kept = []

    for cx, cy, diameter in selected:
        if any(
            (cx - x) ** 2 + (cy - y) ** 2 < 2.0 ** 2
            for x, y, _ in kept
        ):
            continue
        kept.append((cx, cy, diameter))

    return [
        (
            cx * RENDER_SCALE,
            cy * RENDER_SCALE,
            diameter * RENDER_SCALE * 0.58
        )
        for cx, cy, diameter in kept
    ]


def _raster_socket_centers(image_bgr):
    """原樣保留：在平面圖區域使用 OpenCV HoughCircles 定位外圈。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    x0 = int(w * PLAN_X0)
    x1 = int(w * PLAN_X1)
    y0 = int(h * PLAN_Y0)
    y1 = int(h * PLAN_Y1)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    small = cv2.resize(
        roi,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_AREA
    )
    blurred = cv2.GaussianBlur(small, (5, 5), 1.2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=27,
        param1=80,
        param2=39,
        minRadius=10,
        maxRadius=19
    )

    if circles is None:
        return []

    results = []

    for cx, cy, radius in np.round(circles[0]).astype(int):
        results.append((
            float(x0 + cx * 2),
            float(y0 + cy * 2),
            float(max(radius * 2, 12))
        ))

    return results


def _clean_template(image):
    """驗證模板有有效墨線；回傳裁白後的副本，不改動原模板座標。"""
    if image is None:
        return None

    gray = image.copy()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w < 5 or h < 5:
        return None
    return gray[y:y + h, x:x + w]


def load_symbol_templates():
    """載入 0～12 號原始模板；保留原始座標以便圓心錨定。"""
    templates = {}
    missing = []

    for symbol_id in range(13):
        path = SYMBOL_DIR / f"{symbol_id}.png"
        if not path.exists():
            missing.append(f"{symbol_id}.png")
            continue

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or _clean_template(image) is None:
            missing.append(f"{symbol_id}.png")
            continue

        templates[symbol_id] = image

    return templates, missing


def _edge_image(gray):
    """以邊緣特徵降低線條濃淡與掃描灰階差異。"""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blur, 40, 120)


def _extract_center_patch(image_bgr, center, canvas_size):
    """以原紅圈中心擷取固定畫布；不在周圍滑動找最佳位置。"""
    cx, cy, _ = center
    half = canvas_size // 2
    center_x = int(round(cx))
    center_y = int(round(cy))
    x0 = center_x - half
    y0 = center_y - half
    x1 = x0 + canvas_size
    y1 = y0 + canvas_size

    patch = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    source_x0 = max(0, x0)
    source_y0 = max(0, y0)
    source_x1 = min(image_bgr.shape[1], x1)
    source_y1 = min(image_bgr.shape[0], y1)

    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return patch

    source = cv2.cvtColor(
        image_bgr[source_y0:source_y1, source_x0:source_x1],
        cv2.COLOR_BGR2GRAY
    )
    dest_x0 = source_x0 - x0
    dest_y0 = source_y0 - y0
    patch[
        dest_y0:dest_y0 + source.shape[0],
        dest_x0:dest_x0 + source.shape[1]
    ] = source
    return patch


def _masked_edge_f1(template_edges, target_edges, mask,
                    tolerance=CLASSIFY_EDGE_TOLERANCE):
    """只在指定的符號區域內計算雙向邊緣 F1，降低牆線干擾。"""
    template_points = (template_edges > 0) & mask
    target_points = (target_edges > 0) & mask
    template_count = int(np.count_nonzero(template_points))
    target_count = int(np.count_nonzero(target_points))
    if template_count < 4 or target_count < 4:
        return 0.0

    # 距離場只由遮罩內的邊緣建立，遮罩外線條不能替範本加分。
    distance_to_target = cv2.distanceTransform(
        (~target_points).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_template = cv2.distanceTransform(
        (~template_points).astype(np.uint8), cv2.DIST_L2, 3
    )
    precision = float(np.mean(distance_to_target[template_points] <= tolerance))
    recall = float(np.mean(distance_to_template[target_points] <= tolerance))
    return float(2.0 * precision * recall / (precision + recall + 1e-6))


def _masked_ink_f1(template_gray, target_gray, mask, threshold=190):
    """比對圓內黑色筆畫／填色分布，區分實心與空心圖形。"""
    template_ink = (template_gray < threshold) & mask
    target_ink = (target_gray < threshold) & mask
    template_count = int(np.count_nonzero(template_ink))
    target_count = int(np.count_nonzero(target_ink))
    if template_count < 4 or target_count < 4:
        return 0.0
    intersection = int(np.count_nonzero(template_ink & target_ink))
    return float(2.0 * intersection / (template_count + target_count + 1e-6))


def _normalize_glyph(gray):
    """將代碼 ROI 轉成置中的 64x64 二值影像，供 HOG 比對。"""
    if gray is None or gray.size == 0:
        return np.zeros((64, 64), dtype=np.uint8)
    binary = (gray < 190).astype(np.uint8) * 255
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((64, 64), dtype=np.uint8)

    binary = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = binary.shape
    scale = min(52.0 / max(width, 1), 52.0 / max(height, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        binary, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros((64, 64), dtype=np.uint8)
    x0 = (64 - new_width) // 2
    y0 = (64 - new_height) // 2
    canvas[y0:y0 + new_height, x0:x0 + new_width] = resized
    return canvas


def _hog_descriptor():
    """建立 HOG 描述子；只用於範本旁的英文代碼，不做文字辨識。"""
    return cv2.HOGDescriptor(
        (64, 64), (16, 16), (8, 8), (8, 8), 9
    )


_HOG = None
_GLYPH_HOG_TEMPLATES = None


def _get_glyph_hog_templates(templates):
    """依 13 張已配置範本建立英文字形特徵。"""
    global _HOG, _GLYPH_HOG_TEMPLATES
    if _GLYPH_HOG_TEMPLATES is not None:
        return _GLYPH_HOG_TEMPLATES

    _HOG = _hog_descriptor()
    results = {}
    boxes = dict(GLYPH_BOXES)
    boxes.update(INTERNAL_GLYPH_BOXES)
    for symbol_id, box in boxes.items():
        image = templates.get(symbol_id)
        if image is None:
            continue
        x0, y0, x1, y1 = box
        glyph = _normalize_glyph(image[y0:y1, x0:x1])
        vector = _HOG.compute(glyph).flatten().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            vector /= norm
        results[symbol_id] = vector
    _GLYPH_HOG_TEMPLATES = results
    return results


def _template_features(template_gray, symbol_id, target_radius,
                       scale_variant):
    """以範本圓心對齊，擷取圓內圖形遮罩與代碼 ROI 遮罩。"""
    if symbol_id not in TEMPLATE_CIRCLE_GEOMETRY:
        return None

    # 最右側黑線是圖例欄位分隔線，並非圖示筆畫。
    source = template_gray[:, :min(template_gray.shape[1], 60)]
    anchor_x, anchor_y, template_radius = TEMPLATE_CIRCLE_GEOMETRY[symbol_id]
    scale = float(target_radius) * float(scale_variant) / float(template_radius)
    center = CLASSIFY_CANVAS_SIZE // 2

    matrix = cv2.getRotationMatrix2D(
        (anchor_x, anchor_y), 0.0, scale
    )
    matrix[0, 2] += center - anchor_x
    matrix[1, 2] += center - anchor_y

    warped = cv2.warpAffine(
        source, matrix, (CLASSIFY_CANVAS_SIZE, CLASSIFY_CANVAS_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )
    edges = _edge_image(warped)

    yy, xx = np.ogrid[:source.shape[0], :source.shape[1]]
    radial = np.sqrt((xx - anchor_x) ** 2 + (yy - anchor_y) ** 2)
    body_mask_source = (radial < template_radius * 0.72).astype(np.uint8) * 255
    body_mask = cv2.warpAffine(
        body_mask_source, matrix,
        (CLASSIFY_CANVAS_SIZE, CLASSIFY_CANVAS_SIZE),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ) > 0

    glyph_box = INTERNAL_GLYPH_BOXES.get(symbol_id, GLYPH_BOXES.get(symbol_id))
    glyph_mask = None
    glyph_roi = None
    if glyph_box:
        gx0, gy0, gx1, gy1 = glyph_box
        glyph_mask_source = np.zeros(source.shape, dtype=np.uint8)
        pad_template = 1
        glyph_mask_source[
            max(0, gy0 - pad_template):min(source.shape[0], gy1 + pad_template),
            max(0, gx0 - pad_template):min(source.shape[1], gx1 + pad_template)
        ] = 255
        glyph_mask = cv2.warpAffine(
            glyph_mask_source, matrix,
            (CLASSIFY_CANVAS_SIZE, CLASSIFY_CANVAS_SIZE),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        ) > 0

        # HOG 固定 ROI 只看字母應在的位置，沒有自由搜尋或 OCR。
        x_a = center + (gx0 - anchor_x) * scale
        x_b = center + (gx1 - anchor_x) * scale
        y_a = center + (gy0 - anchor_y) * scale
        y_b = center + (gy1 - anchor_y) * scale
        pad_x = max(2, int(round((x_b - x_a) * 0.22)))
        pad_y = max(2, int(round((y_b - y_a) * 0.22)))
        glyph_roi = (
            max(0, int(round(x_a - pad_x))),
            max(0, int(round(y_a - pad_y))),
            min(CLASSIFY_CANVAS_SIZE, int(round(x_b + pad_x))),
            min(CLASSIFY_CANVAS_SIZE, int(round(y_b + pad_y)))
        )

    return edges, body_mask, glyph_mask, glyph_roi, warped


def _score_glyph_hog(patch, glyph_roi, symbol_id, hog_templates):
    if glyph_roi is None or symbol_id not in hog_templates:
        return 0.0
    x0, y0, x1, y1 = glyph_roi
    if x1 <= x0 or y1 <= y0:
        return 0.0
    normalized = _normalize_glyph(patch[y0:y1, x0:x1])
    vector = _HOG.compute(normalized).flatten().astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return 0.0
    vector /= norm
    similarity = float(np.dot(vector, hog_templates[symbol_id]))
    return float(np.clip(similarity, 0.0, 1.0))


def _score_glyph_group_evidence(patch, glyph_roi, hog_templates, symbol_id):
    """以候選的固定代碼區，交叉比較 F/H/G/K/W/J/C/E 字形。

    只讀取依範本定位的局部像素，不辨識一般文字、不使用 OCR，
    也不在紅圈周圍任意搜尋字母。
    """
    if glyph_roi is None or symbol_id not in SYMBOL_GLYPH_GROUP:
        return {}, None, 0.0, 0.0

    x0, y0, x1, y1 = glyph_roi
    if x1 <= x0 or y1 <= y0:
        return {}, None, 0.0, 0.0

    normalized = _normalize_glyph(patch[y0:y1, x0:x1])
    vector = _HOG.compute(normalized).flatten().astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return {}, None, 0.0, 0.0
    vector /= norm

    group_scores = {}
    for group_name, member_ids in GLYPH_GROUPS.items():
        similarities = [
            float(np.dot(vector, hog_templates[member_id]))
            for member_id in member_ids
            if member_id in hog_templates
        ]
        if similarities:
            group_scores[group_name] = float(np.clip(max(similarities), 0.0, 1.0))

    expected_group = SYMBOL_GLYPH_GROUP[symbol_id]
    expected_score = group_scores.get(expected_group, 0.0)
    best_other = max(
        (score for group, score in group_scores.items() if group != expected_group),
        default=0.0
    )
    conflict = float(best_other - expected_score)
    best_group = max(group_scores, key=group_scores.get) if group_scores else None
    return group_scores, best_group, expected_score, conflict


def _score_one_candidate(patch, patch_edges, templates, symbol_id,
                         target_radius, hog_templates):
    """同時比較圓內圖形、指定位置字形邊緣與 HOG 形狀。"""
    best = None
    for scale_variant in CLASSIFY_SCALE_VARIANTS:
        feature = _template_features(
            templates[symbol_id], symbol_id, target_radius, scale_variant
        )
        if feature is None:
            continue
        template_edges, body_mask, glyph_mask, glyph_roi, template_gray = feature
        body_score = _masked_edge_f1(template_edges, patch_edges, body_mask)
        ink_score = _masked_ink_f1(template_gray, patch, body_mask)

        glyph_group_scores = {}
        glyph_group_best = None
        glyph_group_expected = 0.0
        glyph_group_conflict = 0.0
        if glyph_mask is None:
            glyph_score = 0.0
            hog_score = 0.0
            combined = body_score
        else:
            glyph_score = _masked_edge_f1(template_edges, patch_edges, glyph_mask)
            hog_score = _score_glyph_hog(
                patch, glyph_roi, symbol_id, hog_templates
            )
            glyph_group_scores, glyph_group_best, glyph_group_expected, glyph_group_conflict = (
                _score_glyph_group_evidence(
                    patch, glyph_roi, hog_templates, symbol_id
                )
            )
            # 英文代碼的辨別力高於相似圓內圖形，故給較高權重。
            combined = (
                0.15 * body_score
                + 0.40 * glyph_score
                + 0.45 * hog_score
            )

        row = {
            "score": float(combined),
            "body": float(body_score),
            "ink": float(ink_score),
            "glyph": float(glyph_score),
            "hog": float(hog_score),
            "glyph_group_scores": glyph_group_scores,
            "glyph_group_best": glyph_group_best,
            "glyph_group_expected": (
                SYMBOL_GLYPH_GROUP.get(symbol_id)
            ),
            "glyph_group_expected_score": float(glyph_group_expected),
            "glyph_group_conflict": float(glyph_group_conflict),
            "scale": float(scale_variant),
        }
        if best is None or row["score"] > best["score"]:
            best = row
    return best or {"score": 0.0, "body": 0.0, "ink": 0.0,
                    "glyph": 0.0, "hog": 0.0,
                    "glyph_group_scores": {}, "glyph_group_best": None,
                    "glyph_group_expected": SYMBOL_GLYPH_GROUP.get(symbol_id),
                    "glyph_group_expected_score": 0.0,
                    "glyph_group_conflict": 0.0, "scale": 1.0}


def _get_center_template_scores(image_bgr, center, templates,
                                target_radius, hog_templates):
    """逐一比對 1～12 號；以紅圈中心為軸校正四個正交方向。

    施工平面圖中的符號可能依牆面旋轉。這裡只旋轉固定尺寸的局部
    候選圖塊，不改變紅圈中心、不做任意平移搜尋，也不改動定位結果。
    """
    patch = _extract_center_patch(
        image_bgr, center, CLASSIFY_CANVAS_SIZE
    )
    score_map = {}

    # 0/90/180/270 度足以涵蓋平面圖常見的正交牆面方向。
    # np.rot90 保持方形畫布尺寸，旋轉中心就是原紅圈中心。
    for angle, k in ((0, 0), (90, 1), (180, 2), (270, 3)):
        oriented_patch = patch if k == 0 else np.ascontiguousarray(np.rot90(patch, k))
        patch_edges = _edge_image(oriented_patch)

        for symbol_id in range(1, 13):
            template = templates.get(symbol_id)
            if template is None:
                continue
            item = _score_one_candidate(
                oriented_patch, patch_edges, templates, symbol_id,
                target_radius, hog_templates
            )
            item = dict(item)
            item["orientation"] = angle

            previous = score_map.get(symbol_id)
            if previous is None or item["score"] > previous["score"]:
                score_map[symbol_id] = item

    return score_map


def _decide_symbol(score_map):
    """以圖形、代碼形狀及相對位置共同決定類別；證據不足就不猜。"""
    # 非 H 代碼採較嚴格的 HOG 門檻，避免把 CV、TC、牆線或尺寸註記
    # 當成 F/G/K/W/C/E。H 會由 4/9/12 共用，因此另以圓內圖形與填色判斷。
    thresholds = {
        2:  {"score": 0.70, "glyph": 0.64, "hog": 0.72, "body": 0.55, "ink": 0.00},
        4:  {"score": 0.70, "glyph": 0.73, "hog": 0.60, "body": 0.55, "ink": 0.42},
        5:  {"score": 0.70, "glyph": 0.64, "hog": 0.74, "body": 0.55, "ink": 0.00},
        6:  {"score": 0.70, "glyph": 0.56, "hog": 0.70, "body": 0.55, "ink": 0.60},
        7:  {"score": 0.70, "glyph": 0.64, "hog": 0.72, "body": 0.55, "ink": 0.00},
        8:  {"score": 0.73, "glyph": 0.72, "hog": 0.68, "body": 0.58, "ink": 0.00},
        9:  {"score": 0.70, "glyph": 0.73, "hog": 0.60, "body": 0.55, "ink": 0.60},
        10: {"score": 0.70, "glyph": 0.64, "hog": 0.74, "body": 0.55, "ink": 0.48},
        11: {"score": 0.70, "glyph": 0.64, "hog": 0.75, "body": 0.52, "ink": 0.40},
        12: {"score": 0.70, "glyph": 0.73, "hog": 0.60, "body": 0.55, "ink": 0.35},
    }

    eligible = []
    reliable_code_evidence = []
    for symbol_id in CODE_SYMBOL_IDS:
        item = score_map.get(symbol_id)
        if not item:
            continue
        th = thresholds[symbol_id]
        # 若同一位置的字形更像另一種英文字母，排除該類別。
        # 這是「反證」機制，不要求每個候選必須有很大的字母分差；
        # 對 G/C 等近似字形保留整體圖形證據，避免過度漏判。
        code_conflict = float(item.get("glyph_group_conflict", 0.0))
        allowed_conflict = GLYPH_CODE_CONFLICT_MARGIN.get(symbol_id, 0.035)
        code_consistent = code_conflict <= allowed_conflict

        passes = (
            item.get("score", 0.0) >= th["score"]
            and item.get("glyph", 0.0) >= th["glyph"]
            and item.get("hog", 0.0) >= th["hog"]
            and item.get("body", 0.0) >= th["body"]
            and item.get("ink", 0.0) >= th["ink"]
            and code_consistent
        )
        if passes:
            eligible.append((item["score"], symbol_id, item))
            reliable_code_evidence.append(symbol_id)

    if eligible:
        eligible.sort(key=lambda x: x[0], reverse=True)
        best_score, best_id, best_item = eligible[0]
        second_score = eligible[1][0] if len(eligible) > 1 else 0.0
        margin = best_score - second_score
        if margin >= CLASSIFY_CODE_MIN_MARGIN:
            # J 與 W 的鄰接筆畫有時會交疊；兩者都有明確證據時保留待確認。
            j_detail = score_map.get(8)
            if (
                best_id == 7 and j_detail
                and j_detail.get("glyph", 0.0) >= 0.60
                and j_detail.get("hog", 0.0) >= 0.55
                and j_detail.get("body", 0.0) >= 0.45
            ):
                return None, best_score, margin, best_item, "possible_internal_J"
            return best_id, best_score, margin, best_item, "glyph_visual_rotated"
        return None, best_score, margin, best_item, "ambiguous_code"

    # 有足夠完整的代碼證據但候選彼此打平時，不退回 1/3 號猜測。
    if reliable_code_evidence:
        top_item = max(
            (score_map[sid] for sid in reliable_code_evidence),
            key=lambda x: x.get("score", 0.0)
        )
        return None, top_item["score"], 0.0, top_item, "code_needs_review"

    # 1、3 號沒有英文字母。只有在圓內結構相似度很高，且沒有可靠代碼時才使用。
    body_ranked = sorted(
        ((score_map[sid]["body"], sid, score_map[sid])
         for sid in BODY_ONLY_SYMBOL_IDS if sid in score_map),
        reverse=True,
        key=lambda x: x[0]
    )
    if not body_ranked:
        return None, 0.0, 0.0, {}, "no_match"

    best_score, best_id, best_item = body_ranked[0]
    second_score = body_ranked[1][0] if len(body_ranked) > 1 else 0.0
    margin = best_score - second_score
    if (
        best_score >= CLASSIFY_BODY_ONLY_MIN_SCORE
        and margin >= CLASSIFY_BODY_ONLY_MIN_MARGIN
    ):
        return best_id, best_score, margin, best_item, "body_visual_rotated"
    return None, best_score, margin, best_item, "needs_review"


def _classify_one_center(image_bgr, center, templates,
                         radius_scale=1.0, hog_templates=None):
    """針對單一中心分類；沿用紅圈定位座標，只調整分類結果。"""
    radius = float(center[2]) * float(radius_scale)
    radius = float(np.clip(radius, 20.0, 36.0))
    if hog_templates is None:
        hog_templates = _get_glyph_hog_templates(templates)
    score_map = _get_center_template_scores(
        image_bgr, center, templates, radius, hog_templates
    )
    symbol_id, score, margin, detail, decision = _decide_symbol(score_map)
    return symbol_id, score, margin, detail, decision


def _classify_centers(image_bgr, centers, templates, radius_scale=1.0):
    """每個紅圈以自己的實際半徑比對圖示與代碼；不修改定位清單。"""
    counts = {i: 0 for i in range(13)}
    classified = []
    unknown = 0
    hog_templates = _get_glyph_hog_templates(templates)

    for center in centers:
        symbol_id, score, margin, detail, decision = _classify_one_center(
            image_bgr, center, templates,
            radius_scale=radius_scale,
            hog_templates=hog_templates
        )
        if symbol_id is None:
            unknown += 1
        else:
            counts[symbol_id] += 1

        classified.append({
            "center": center,
            "symbol_id": symbol_id,
            "score": float(score),
            "margin": float(margin),
            "decision": decision,
            "features": detail,
        })

    glyph_matched = sum(
        1 for item in classified
        if item["symbol_id"] not in (None, 1, 3)
    )
    return counts, unknown, classified, glyph_matched


def _analyze_pdf(file_bytes, page_number, templates):
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"PDF 無法開啟：{exc}")

    try:
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(document):
            raise ValueError(
                f"PDF 只有 {len(document)} 頁，指定的第 {page_number} 頁不存在。"
            )

        page = document.load_page(page_index)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE),
            alpha=False,
            colorspace=fitz.csRGB
        )

        rgb = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(pix.height, pix.width, pix.n)

        image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 核心偵測方式保持原樣：先向量圓形，沒有結果才用 Hough 圓形。
        vector_centers = _vector_socket_centers(page)
        if vector_centers:
            centers = vector_centers
            radius_scale = 1.0 / 1.16
            method = "PDF 向量符號定位"
            note = "保留原本向量圓形路徑定位方式；分類以圓心固定對位。"
        else:
            centers = _raster_socket_centers(image_bgr)
            radius_scale = 1.0
            method = "高解析度圖面圓形辨識"
            note = "保留原本 HoughCircles 圓形定位方式；分類以圓心固定對位。"

        if not centers:
            raise ValueError(
                f"PDF 第 {page_number} 頁沒有找到可辨識的插座外圈，請確認頁碼。"
            )

        counts, unknown, classified, glyph_matched = _classify_centers(
            image_bgr, centers, templates, radius_scale=radius_scale
        )

        return {
            "count": len(centers),
            "counts": counts,
            "unknown": unknown,
            "glyph_matched": glyph_matched,
            "image": _jpeg_data_url(_mark_centers(image_bgr, centers)),
            "method": method,
            "note": note + "；分類使用圖示內部結構與英文字母代碼的影像特徵比對，並自動校正 0°／90°／180°／270°方向；英文代碼會再與 F／H／G／K／W／J／C／E 字形群組交叉比對，不使用 OCR。"
        }
    finally:
        document.close()


def _analyze_image(file_bytes, templates):
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("圖片無法讀取。")

    h, w = image.shape[:2]
    if max(h, w) < 1600:
        raise ValueError(
            "圖片解析度太低，無法可靠清點小型設備符號。請上傳原始 PDF。"
        )

    centers = _raster_socket_centers(image)
    if not centers:
        raise ValueError("圖片中沒有找到可辨識的插座外圈。")

    counts, unknown, classified, glyph_matched = _classify_centers(
        image, centers, templates, radius_scale=1.0
    )

    return {
        "count": len(centers),
        "counts": counts,
        "unknown": unknown,
        "glyph_matched": glyph_matched,
        "image": _jpeg_data_url(_mark_centers(image, centers)),
        "method": "高解析度圖面圓形辨識",
        "note": "保留原本 HoughCircles 圓形定位方式；分類使用圖示與英文字母代碼影像特徵，不使用 OCR。"
    }


def analyze_upload(upload, page_number, templates):
    filename = upload.filename or ""
    ext = _filename_ext(filename)

    if ext not in ALLOWED:
        raise ValueError("只支援 PDF、PNG、JPG、JPEG、BMP、TIF、TIFF。")

    raw = upload.read()
    if not raw:
        raise ValueError(f"檔案是空的：{filename}")

    if ext == "pdf":
        return _analyze_pdf(raw, page_number, templates)

    return _analyze_image(raw, templates)


def _build_rows(old_result, new_result):
    rows = []

    for symbol_id in range(13):
        old_value = int(old_result["counts"].get(symbol_id, 0))
        new_value = int(new_result["counts"].get(symbol_id, 0))
        diff = new_value - old_value

        rows.append({
            "id": symbol_id,
            "name": SYMBOL_NAMES[symbol_id],
            "old_count": old_value,
            "new_count": new_value,
            "added_count": max(diff, 0),
            "removed_count": max(-diff, 0),
            "difference": diff
        })

    old_unknown = int(old_result["unknown"])
    new_unknown = int(new_result["unknown"])
    unknown_diff = new_unknown - old_unknown

    if old_unknown or new_unknown:
        rows.append({
            "id": "unknown",
            "name": "未分類（需人工核對）",
            "old_count": old_unknown,
            "new_count": new_unknown,
            "added_count": max(unknown_diff, 0),
            "removed_count": max(-unknown_diff, 0),
            "difference": unknown_diff
        })

    return rows


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/analyze")
def analyze_api():
    old_file = request.files.get("old")
    new_file = request.files.get("new")

    if not old_file or not new_file:
        return jsonify(error="請同時選擇原始圖與設計師修改圖。"), 400

    try:
        old_page = int(request.form.get("old_page", "4"))
        new_page = int(request.form.get("new_page", "3"))

        if old_page < 1 or new_page < 1:
            raise ValueError("PDF 頁碼必須從 1 開始。")

        templates, missing = load_symbol_templates()
        if missing:
            raise ValueError(
                "找不到完整的 13 種符號模板，請確認以下檔案存在於 "
                "static/symbols/ 資料夾："
                + "、".join(missing)
            )

        old_result = analyze_upload(old_file, old_page, templates)
        new_result = analyze_upload(new_file, new_page, templates)

        rows = _build_rows(old_result, new_result)

        old_total = old_result["count"]
        new_total = new_result["count"]
        difference = new_total - old_total

        added_total = max(difference, 0)
        removed_total = max(-difference, 0)

        old_classified = sum(old_result["counts"].values())
        new_classified = sum(new_result["counts"].values())

        row_source = (
            "分類以紅圈中心固定對位，並使用每個候選自己的圓徑調整模板。"
            "圓內圖形及 F/G/K/W/J/C/E/H 等英文字母代碼均以影像形狀比對，並與其他字形群組交叉檢查；同時比對 0°／90°／180°／270°方向，不使用 OCR。"
            "若代碼、圓內圖形或候選差距不足以唯一確認，會保留為未分類，不強制猜測。"
            "不讀取右側圖例／數量表，也不以整頁像素差異代替設備辨識。"
            "0 號範本為長方形設備符號，與既有紅圈定位不同；本版仍不把圓形候選誤判為 0 號，"
            "0 號需另加獨立長方形符號定位器。"
        )

        if old_result["unknown"] or new_result["unknown"]:
            row_source += (
                f" 未分類數量：原圖 {old_result['unknown']} 個、"
                f"修改後 {new_result['unknown']} 個；"
                "這些位置不會被強制歸入 13 種類別，請人工核對。"
            )

        return jsonify(
            old_detected_count=old_total,
            new_detected_count=new_total,
            difference=difference,
            added_count=added_total,
            removed_count=removed_total,
            old_classified_count=old_classified,
            new_classified_count=new_classified,
            old_unknown_count=old_result["unknown"],
            new_unknown_count=new_result["unknown"],
            old_glyph_matched_count=old_result.get("glyph_matched", 0),
            new_glyph_matched_count=new_result.get("glyph_matched", 0),
            old_image=old_result["image"],
            new_image=new_result["image"],
            old_method=old_result["method"],
            new_method=new_result["method"],
            old_note=old_result["note"],
            new_note=new_result["note"],
            row_source=row_source,
            rows=rows
        )

    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        app.logger.exception("DrawDiff analysis failed")
        return jsonify(error=f"辨識失敗：{exc}"), 500


@app.post("/export")
def export_csv():
    data = request.get_json(silent=True) or {}
    rows = data.get("rows", [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "設備編號",
        "設備項目",
        "原圖數量",
        "修改後數量",
        "增加",
        "減少",
        "淨增減"
    ])

    old_total = 0
    new_total = 0
    added_total = 0
    removed_total = 0

    for row in rows:
        old_value = int(row.get("old_count", 0))
        new_value = int(row.get("new_count", 0))
        diff = new_value - old_value

        old_total += old_value
        new_total += new_value
        added_total += max(diff, 0)
        removed_total += max(-diff, 0)

        writer.writerow([
            row.get("id", ""),
            row.get("name", ""),
            old_value,
            new_value,
            max(diff, 0),
            max(-diff, 0),
            diff
        ])

    writer.writerow([
        "",
        "合計（含未分類）",
        old_total,
        new_total,
        added_total,
        removed_total,
        new_total - old_total
    ])

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                "attachment; filename=DrawDiff_13種設備增減統計.csv"
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

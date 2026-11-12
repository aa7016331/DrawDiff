
import os
import csv
import io
import base64
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
from flask import Flask, render_template_string, request, jsonify, Response

BASE = Path(__file__).resolve().parent
ALLOWED = {"pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024

INDEX_HTML = r'''
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DrawDiff｜插座增減辨識</title>
<style>
:root{font-family:Arial,"Noto Sans TC",sans-serif;color:#152432;background:#f2f5f7}
*{box-sizing:border-box}
body{margin:0;padding:24px}
main{max-width:1420px;margin:0 auto}
header{margin:0 auto 20px;max-width:1420px}
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
#status{margin-top:14px;min-height:20px;color:#5d6c77;font-size:14px}
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
table{border-collapse:collapse;width:100%;margin-top:20px}
th,td{border-bottom:1px solid #e3e9ed;padding:12px;text-align:center}
th{background:#edf2f5}
td:first-child,th:first-child{text-align:left}
.note{color:#687987;font-size:13px;margin-top:12px;line-height:1.5}
.symbol-cell{display:flex;align-items:center;gap:10px;text-align:left;min-width:260px}.symbol-cell img{width:48px;height:44px;object-fit:contain;background:#fff;border:1px solid #e5eaee;border-radius:5px;padding:2px}.positive{color:#13724e;font-weight:700}.negative{color:#b42318;font-weight:700}.total-row{font-weight:800;background:#f1f5f7}.total-row td{border-top:2px solid #cbd6dd}
@media(max-width:850px){body{padding:12px}.upload-grid,.compare-grid{grid-template-columns:1fr}.main-result{grid-template-columns:1fr 1fr}.card{padding:16px}}
</style>
</head>
<body>
<header>
  <h1>DrawDiff｜插座增減辨識</h1>
  <div class="sub">逐項列出 17 種電氣／弱電插座與電源圖例的原圖數量、修改後數量、增加及減少；左右紅圈用來核對平面圖位置。</div>
</header>
<main>
<section class="card">
  <h2>上傳圖面</h2>
  <form id="uploadForm">
    <div class="upload-grid">
      <div class="upload-box">
        <strong>原始圖</strong>
        <div class="sub">上傳原始 PDF，預設讀取第 4 頁</div>
        <input id="oldFile" name="old" type="file" accept=".pdf,application/pdf" required>
        <div class="page-row">PDF 頁碼 <input id="oldPage" name="old_page" type="number" min="1" value="4"></div>
      </div>
      <div class="upload-box">
        <strong>設計師修改圖</strong>
        <div class="sub">上傳設計師修改 PDF，預設讀取第 3 頁</div>
        <input id="newFile" name="new" type="file" accept=".pdf,application/pdf" required>
        <div class="page-row">PDF 頁碼 <input id="newPage" name="new_page" type="number" min="1" value="3"></div>
      </div>
    </div>
    <div class="actions">
      <button class="primary" id="analyzeButton" type="submit">開始辨識插座數量</button>
      <button class="secondary" id="exportButton" type="button" disabled>匯出增減表 CSV</button>
    </div>
  </form>
  <div id="status"></div>
</section>

<section class="card" id="result">
  <h2>插座增減結果</h2>
  <div class="summary" id="summary"></div>
  <div class="main-result">
    <div class="metric"><div class="label">原始圖插座總數</div><div class="value" id="oldCount">—</div></div>
    <div class="metric"><div class="label">設計師修改圖插座總數</div><div class="value" id="newCount">—</div></div>
    <div class="metric"><div class="label">淨增減</div><div class="value" id="netCount">—</div></div>
  </div>
  <div class="compare-grid">
    <div class="image-panel">
      <div class="image-title">原始圖（紅圈為辨識到的插座）</div>
      <img id="oldResult" alt="原始圖插座辨識結果">
      <div class="method" id="oldMethod"></div>
    </div>
    <div class="image-panel">
      <div class="image-title">設計師修改圖（紅圈為辨識到的插座）</div>
      <img id="newResult" alt="修改圖插座辨識結果">
      <div class="method" id="newMethod"></div>
    </div>
  </div>
  <table>
    <thead><tr><th>圖例／設備項目</th><th>原圖數量</th><th>修改後數量</th><th>增加</th><th>減少</th><th>淨增減</th></tr></thead>
    <tbody id="resultRows"></tbody>
  </table>
  <div class="note" id="resultNote"></div>
</section>
</main>
<script>
let latestRows=[];
const form=document.getElementById('uploadForm');
const statusEl=document.getElementById('status');
const analyzeButton=document.getElementById('analyzeButton');
const exportButton=document.getElementById('exportButton');

form.addEventListener('submit',async (event)=>{
  event.preventDefault();
  const oldFile=document.getElementById('oldFile').files[0];
  const newFile=document.getElementById('newFile').files[0];
  if(!oldFile||!newFile){statusEl.textContent='請同時選擇原始圖及修改後圖。';statusEl.className='error';return;}
  const payload=new FormData(form);
  analyzeButton.disabled=true;
  exportButton.disabled=true;
  statusEl.textContent='正在讀取指定頁面、辨識平面圖位置及逐項圖例數量。';
  statusEl.className='';
  try{
    const response=await fetch('/analyze',{method:'POST',body:payload});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||'辨識失敗');
    document.getElementById('result').style.display='block';
    document.getElementById('oldCount').textContent=data.old_count;
    document.getElementById('newCount').textContent=data.new_count;
    const net=data.difference;
    document.getElementById('netCount').textContent=(net>0?'+':'')+net;
    const summary=document.getElementById('summary');
    summary.innerHTML='';
    [
      ['原圖 '+data.old_count+' 個',''],
      ['修改後 '+data.new_count+' 個',''],
      ['增加 '+data.added_count+' 個','plus'],
      ['減少 '+data.removed_count+' 個','minus']
    ].forEach(([label,cls])=>{const d=document.createElement('div');d.className='pill '+cls;d.textContent=label;summary.appendChild(d);});
    document.getElementById('oldResult').src=data.old_image;
    document.getElementById('newResult').src=data.new_image;
    document.getElementById('oldMethod').textContent=data.old_method+'。'+data.old_note;
    document.getElementById('newMethod').textContent=data.new_method+'。'+data.new_note;
    const body=document.getElementById('resultRows');
    body.replaceChildren();
    latestRows=data.rows||[];
    latestRows.forEach(row=>{
      const tr=document.createElement('tr');
      const item=document.createElement('td');
      item.className='symbol-cell';
      if(row.icon){const img=document.createElement('img');img.src=row.icon;img.alt='';item.appendChild(img);}
      const name=document.createElement('span');name.textContent=row.name;item.appendChild(name);tr.appendChild(item);
      [row.old_count,row.new_count,row.added_count,row.removed_count,(row.difference>0?'+':'')+row.difference].forEach((v,i)=>{
        const td=document.createElement('td');td.textContent=v;
        if(i===2 && row.added_count>0)td.className='positive';
        if(i===3 && row.removed_count>0)td.className='negative';
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    const totalTr=document.createElement('tr');
    totalTr.className='total-row';
    const totalLabel=document.createElement('td');
    totalLabel.textContent='合計';
    totalTr.appendChild(totalLabel);
    [data.old_count,data.new_count,data.added_count,data.removed_count,(data.difference>0?'+':'')+data.difference].forEach(v=>{
      const td=document.createElement('td');td.textContent=v;totalTr.appendChild(td);
    });
    body.appendChild(totalTr);
    document.getElementById('resultNote').textContent=data.row_source;
    exportButton.disabled=false;
    statusEl.textContent='17 項圖例數量已逐項整理完成。請核對圖面紅圈與下方表格。';
    statusEl.className='';
    document.getElementById('result').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(err){
    statusEl.textContent=err.message;
    statusEl.className='error';
  }finally{
    analyzeButton.disabled=false;
  }
});

exportButton.addEventListener('click',async ()=>{
  if(!latestRows.length)return;
  try{
    const r=await fetch('/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rows:latestRows})});
    if(!r.ok)throw new Error('CSV 匯出失敗');
    const blob=await r.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download='DrawDiff插座增減統計.csv';document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);
  }catch(err){statusEl.textContent=err.message;statusEl.className='error';}
});
</script>
</body>
</html>
'''

# 平面圖部分負責定位插座外圈並標紅圈；逐項數量由設計師頁右側已填圖例表讀取。
# 頁面座標比例與圖例表格座標適用於目前這組版型（原圖第4頁／設計師圖第3頁）。
PLAN_X0 = 0.11
PLAN_X1 = 0.73
PLAN_Y0 = 0.20
PLAN_Y1 = 0.84
RENDER_SCALE = 4.0
LEGEND_OCR_SCALE = 5.0

# 原圖第4頁與設計師修改圖第3頁的插座／弱電圖例順序。
# row_y 是設計師圖第3頁右側圖例表在 5 倍渲染裁切區中的列中心。
# 標配數量 = 原圖數量；客變數量 = 修改後數量。
LEGEND_ITEMS = [
    ("馬桶專用插座 H:30cm", 431),
    ("抽油煙機插座 H:天花板內", 517),
    ("雙聯插座 H:30cm，電視牆 H:80cm", 603),
    ("雙聯插座 H:110cm", 689),
    ("洗碗機專用插座 H:45cm", 774),
    ("微波烤箱專用插座 220V (H:55cm)", 858),
    ("熱水器用插座（防雨型）H:120cm", 944),
    ("廚房櫥櫃燈用電源出線口 H:170cm", 1030),
    ("暖風乾燥機電源出口 220V (H:天花)", 1116),
    ("雙聯紅色緊急插座（弱電箱用、箱體內）", 1243),
    ("雙聯紅色緊急插座（冰箱 H:30cm、電視牆 H:80cm）", 1413),
    ("冷氣電源出口 220V (H:天花板)", 1542),
    ("專迴 220V", 1628),
    ("專迴 110V", 1714),
    ("電話、資訊插座 H:30cm（鄰床頭櫃 H:65cm）", 2254),
    ("電視、資訊插座 H:80cm", 2403),
    ("資訊插座", 2484),
]


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


def _mark_centers(image_bgr, centers, vector_scale=None):
    """在插座符號中心畫紅圈；centers 為 (x, y, radius)。"""
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
    針對 CAD 向量 PDF：插座外圈是重複的四段 Bezier 圓形路徑。
    先限制在平面圖區域，再找出重複最多的圓形直徑群組。
    圖例表格位於右側，因此不會把右側圖例中的圓形樣板納入。
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

        # get_drawings 使用未旋轉座標，轉成頁面實際顯示座標。
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

        # 只保留左側平面圖中的符號，不計右側圖例、標題欄及表格。
        if not (PLAN_X0 * page_w <= cx <= PLAN_X1 * page_w):
            continue
        if not (PLAN_Y0 * page_h <= cy <= PLAN_Y1 * page_h):
            continue

        candidates.append((cx, cy, diameter))

    if len(candidates) < 8:
        return []

    # 以最常出現的圓形尺寸作為插座外圈尺寸。
    # 尺寸四捨五入到 0.1 pt，避免浮點誤差造成分組分裂。
    from collections import Counter
    size_counts = Counter(round(item[2], 1) for item in candidates)
    dominant_size, dominant_count = size_counts.most_common(1)[0]

    if dominant_count < 8:
        return []

    selected = [
        item for item in candidates
        if abs(item[2] - dominant_size) <= 0.25
    ]

    # 圖紙頁面可能包含重複的相同圓形路徑；以中心點去重。
    selected.sort(key=lambda item: item[2], reverse=True)
    kept = []
    for cx, cy, diameter in selected:
        if any((cx - x) ** 2 + (cy - y) ** 2 < 2.0 ** 2 for x, y, _ in kept):
            continue
        kept.append((cx, cy, diameter))

    return [
        (cx * RENDER_SCALE, cy * RENDER_SCALE, diameter * RENDER_SCALE * 0.58)
        for cx, cy, diameter in kept
    ]


def _raster_socket_centers(image_bgr):
    """
    針對沒有向量路徑的高解析 PDF 掃描頁：
    在平面圖 ROI 內偵測插座外圈。參數以目前提供的設計師第 3 頁
    4 倍渲染結果校正；較低解析度圖片會回報辨識限制。
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    x0 = int(w * PLAN_X0)
    x1 = int(w * PLAN_X1)
    y0 = int(h * PLAN_Y0)
    y1 = int(h * PLAN_Y1)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    # 標準 PDF 4 倍渲染時，縮小一半後插座外圈半徑約 10–19 px。
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
        # Hough 在縮小一半的 ROI 上運算，還原至原頁面像素座標。
        results.append((
            float(x0 + cx * 2),
            float(y0 + cy * 2),
            float(max(radius * 2, 12))
        ))

    return results


def _analyze_pdf(file_bytes, page_number):
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
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        vector_centers = _vector_socket_centers(page)
        if vector_centers:
            centers = vector_centers
            method = "PDF 向量符號定位"
            confidence_note = "此頁保留向量圖元，以重複的插座外圈路徑計數。"
        else:
            centers = _raster_socket_centers(image_bgr)
            method = "高解析度圖面圓形辨識"
            confidence_note = (
                "此頁為影像圖面，使用圓形符號定位；請核對紅圈是否都落在插座上。"
            )

        if not centers:
            raise ValueError(
                "這一頁沒有找到可辨識的插座符號。請確認頁碼，或提供更清晰的 PDF。"
            )

        marked = _mark_centers(image_bgr, centers)
        return {
            "count": len(centers),
            "image": _jpeg_data_url(marked),
            "method": method,
            "note": confidence_note
        }
    finally:
        document.close()


def _analyze_image(file_bytes):
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("圖片無法讀取。")

    h, w = image.shape[:2]
    # 低解析度圖片不會因放大而產生新細節。
    if max(h, w) < 1600:
        raise ValueError(
            "這張圖片解析度太低，無法可靠清點小型插座符號。請上傳原始 PDF，並選擇正確頁碼。"
        )

    centers = _raster_socket_centers(image)
    if not centers:
        raise ValueError("圖片中沒有找到可辨識的插座符號。")

    marked = _mark_centers(image, centers)
    return {
        "count": len(centers),
        "image": _jpeg_data_url(marked),
        "method": "高解析度圖面圓形辨識",
        "note": "此頁為點陣圖面，請核對紅圈是否都落在插座上。"
    }


def _ocr_single_quantity(gray_page, x_center, y_center, item_name, column_name):
    """Read one printed quantity from the designer's fixed legend table cell."""
    x0 = max(0, int(x_center - 40))
    x1 = min(gray_page.shape[1], int(x_center + 40))
    y0 = max(0, int(y_center - 42))
    y1 = min(gray_page.shape[0], int(y_center + 42))
    crop = gray_page[y0:y1, x0:x1]

    if crop.size == 0:
        raise ValueError(f"圖例「{item_name}」的{column_name}欄位超出頁面範圍。")

    crop = cv2.resize(crop, (320, 336), interpolation=cv2.INTER_CUBIC)

    # First pass is tuned for the scanned paper form. The second pass handles
    # light/handwritten marks around the printed digit (e.g. the TV/weak-current rows).
    candidates = []
    for threshold in (180, 200):
        processed = cv2.threshold(
            crop,
            threshold,
            255,
            cv2.THRESH_BINARY
        )[1]
        processed = cv2.copyMakeBorder(
            processed,
            50, 50, 50, 50,
            cv2.BORDER_CONSTANT,
            value=255
        )
        text = pytesseract.image_to_string(
            processed,
            config="--psm 6 -c tessedit_char_whitelist=0123456789"
        )
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits and len(digits) <= 2:
            candidates.append(digits)

    if not candidates:
        raise ValueError(
            f"無法讀取圖例「{item_name}」的{column_name}數量。"
            "請確認設計師修改圖右側圖例表的數量欄清楚且已填寫。"
        )

    # Prefer the most frequent reading; on a tie, prefer the shorter numeric
    # token to avoid joining a stray pencil stroke to the printed digit.
    from collections import Counter
    counts = Counter(candidates)
    best = sorted(counts.items(), key=lambda item: (-item[1], len(item[0])))[0][0]
    value = int(best)
    if value < 0 or value > 99:
        raise ValueError(f"圖例「{item_name}」的{column_name}數量讀取異常：{best}")
    return value

def _extract_legend_rows(pdf_bytes, page_number):
    """Read the per-item baseline and customer-change quantity columns."""
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"設計師 PDF 無法開啟：{exc}")

    try:
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(document):
            raise ValueError(f"設計師 PDF 只有 {len(document)} 頁，指定的第 {page_number} 頁不存在。")

        page = document.load_page(page_index)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(LEGEND_OCR_SCALE, LEGEND_OCR_SCALE),
            alpha=False,
            colorspace=fitz.csGRAY
        )
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

        # These positions are relative to the 5x rendered 5、7、9F-A form.
        # The symbol and description columns are to the left of these two count columns.
        standard_x = 5410
        customer_x = 5624
        table_crop_y = 400

        rows = []
        for index, (name, row_y) in enumerate(LEGEND_ITEMS):
            y = table_crop_y + row_y
            old_count = _ocr_single_quantity(
                gray, standard_x, y, name, "標配數量"
            )
            new_count = _ocr_single_quantity(
                gray, customer_x, y, name, "客變數量"
            )
            difference = new_count - old_count

            # Crop the matching symbol from the legend so the results table
            # visually shows the same icon as the paper schedule.
            ix0 = 4515
            ix1 = 4705
            iy0 = max(0, y - 40)
            iy1 = min(gray.shape[0], y + 40)
            icon_crop = gray[iy0:iy1, ix0:ix1]
            ok, icon_buffer = cv2.imencode(".png", icon_crop)
            icon_data = None
            if ok:
                icon_data = (
                    "data:image/png;base64,"
                    + base64.b64encode(icon_buffer.tobytes()).decode("ascii")
                )

            rows.append({
                "id": index + 1,
                "name": name,
                "old_count": old_count,
                "new_count": new_count,
                "added_count": max(difference, 0),
                "removed_count": max(-difference, 0),
                "difference": difference,
                "icon": icon_data
            })

        return rows
    finally:
        document.close()


def analyze_upload(upload, page_number):
    filename = upload.filename or ""
    ext = _filename_ext(filename)
    if ext not in ALLOWED:
        raise ValueError("只支援 PDF、PNG、JPG、JPEG、BMP、TIF、TIFF。")

    raw = upload.read()
    if not raw:
        raise ValueError(f"檔案是空的：{filename}")

    if ext == "pdf":
        return _analyze_pdf(raw, page_number)
    return _analyze_image(raw)


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/analyze")
def analyze_api():
    old_file = request.files.get("old")
    new_file = request.files.get("new")

    if not old_file or not new_file:
        return jsonify(error="請同時選擇原始圖 PDF 與設計師修改圖 PDF。"), 400

    try:
        old_page = int(request.form.get("old_page", "4"))
        new_page = int(request.form.get("new_page", "3"))
        if old_page < 1 or new_page < 1:
            raise ValueError("PDF 頁碼必須從 1 開始。")

        old_raw = old_file.read()
        new_raw = new_file.read()
        if not old_raw or not new_raw:
            raise ValueError("請確認原始圖與設計師修改圖都有正確上傳。")
        if _filename_ext(old_file.filename) != "pdf" or _filename_ext(new_file.filename) != "pdf":
            raise ValueError("逐項插座表需要 PDF 圖面，請上傳原始 PDF 與設計師修改 PDF。")

        old_result = _analyze_pdf(old_raw, old_page)
        new_result = _analyze_pdf(new_raw, new_page)
        rows = _extract_legend_rows(new_raw, new_page)

        old_count = sum(row["old_count"] for row in rows)
        new_count = sum(row["new_count"] for row in rows)
        added_total = sum(row["added_count"] for row in rows)
        removed_total = sum(row["removed_count"] for row in rows)
        difference = new_count - old_count

        mismatch_notes = []
        if old_result["count"] != old_count:
            mismatch_notes.append(
                f"原圖平面紅圈數 {old_result['count']}，圖例數量合計 {old_count}"
            )
        if new_result["count"] != new_count:
            mismatch_notes.append(
                f"修改圖平面紅圈數 {new_result['count']}，圖例數量合計 {new_count}"
            )

        row_source = (
            f"17 項數量逐列讀取設計師 PDF 第 {new_page} 頁右側已填寫的「標配數量／客變數量」欄，"
            "再逐項計算增加、減少及淨增減；沒有把總數平均分配。"
            "平面圖紅圈目前用於核對插座總數與位置，並非將每個紅圈自動分類成 17 種圖例。"
        )
        if mismatch_notes:
            row_source += " 注意：" + "；".join(mismatch_notes) + "，請人工核對。"
        else:
            row_source += " 逐項加總與平面圖紅圈定位總數一致。"

        return jsonify(
            old_count=old_count,
            new_count=new_count,
            difference=difference,
            added_count=added_total,
            removed_count=removed_total,
            old_image=old_result["image"],
            new_image=new_result["image"],
            old_method=old_result["method"],
            new_method=new_result["method"],
            old_note=old_result["note"],
            new_note=new_result["note"],
            old_detected_count=old_result["count"],
            new_detected_count=new_result["count"],
            row_source=row_source,
            rows=rows
        )

    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except pytesseract.pytesseract.TesseractNotFoundError:
        return jsonify(
            error="伺服器缺少 Tesseract OCR。請在 Dockerfile 安裝 tesseract-ocr，並在 requirements.txt 加入 pytesseract。"
        ), 500
    except Exception as exc:
        app.logger.exception("socket count failed")
        return jsonify(error=f"辨識失敗：{exc}"), 500


@app.post("/export")
def export_csv():
    data = request.get_json(silent=True) or {}
    rows = data.get("rows", [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["圖例／設備項目", "原圖數量", "修改後數量", "增加", "減少", "淨增減"])

    old_total = 0
    new_total = 0
    added_total = 0
    removed_total = 0
    for row in rows:
        diff = int(row.get("difference", 0))
        old_value = int(row.get("old_count", 0))
        new_value = int(row.get("new_count", 0))
        added_value = max(diff, 0)
        removed_value = max(-diff, 0)
        old_total += old_value
        new_total += new_value
        added_total += added_value
        removed_total += removed_value
        writer.writerow([
            row.get("name", ""),
            old_value,
            new_value,
            added_value,
            removed_value,
            diff
        ])

    writer.writerow([
        "合計",
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
            "Content-Disposition": "attachment; filename=DrawDiff插座增減統計.csv"
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

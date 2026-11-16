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

# 模板比對參數。低於門檻的結果不強制分類，會列入「未分類」。
CLASSIFY_MIN_SCORE = 0.38
CLASSIFY_MIN_MARGIN = 0.025
CLASSIFY_SCALES = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15)

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
  <div class="sub">保留原本紅圈定位方式，再依 0～12 號符號模板分類，分別統計原圖與修改後圖面。</div>
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
      ['修改後未分類 '+data.new_unknown_count+' 個','']
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
    """裁掉模板白邊，保留符號本體。"""
    if image is None:
        return None

    gray = image.copy()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # 去除極淡掃描底色，將符號統一成黑線、白底。
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w < 5 or h < 5:
        return None

    crop = gray[y:y + h, x:x + w]
    return crop


def load_symbol_templates():
    templates = {}
    missing = []

    for symbol_id in range(13):
        path = SYMBOL_DIR / f"{symbol_id}.png"
        if not path.exists():
            missing.append(f"{symbol_id}.png")
            continue

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        cleaned = _clean_template(image)
        if cleaned is None:
            missing.append(f"{symbol_id}.png")
            continue

        templates[symbol_id] = cleaned

    return templates, missing


def _edge_image(gray):
    # 以線條輪廓比對，降低原圖與修改圖灰階、列印濃淡差異。
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blur, 45, 130)


def _best_template_score(patch_gray, template_gray):
    """
    在單一紅圈附近，以多尺度及四個直角方向比對模板。
    只負責分類，不參與紅圈中心偵測。
    """
    patch_edge = _edge_image(patch_gray)
    template_edge = _edge_image(template_gray)

    best = -1.0
    for angle in (0, 90, 180, 270):
        if angle == 0:
            rotated = template_edge
        elif angle == 90:
            rotated = cv2.rotate(template_edge, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(template_edge, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(template_edge, cv2.ROTATE_90_COUNTERCLOCKWISE)

        th, tw = rotated.shape
        for scale in CLASSIFY_SCALES:
            nw = max(5, int(tw * scale))
            nh = max(5, int(th * scale))

            if nw >= patch_edge.shape[1] or nh >= patch_edge.shape[0]:
                continue

            resized = cv2.resize(
                rotated,
                (nw, nh),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            )

            if np.std(resized) < 2 or np.std(patch_edge) < 2:
                continue

            response = cv2.matchTemplate(
                patch_edge,
                resized,
                cv2.TM_CCOEFF_NORMED
            )
            _, score, _, _ = cv2.minMaxLoc(response)
            if np.isfinite(score):
                best = max(best, float(score))

    return best


def _classify_one_center(image_bgr, center, templates):
    cx, cy, radius = center
    h, w = image_bgr.shape[:2]

    # 紅圈中心周圍取局部圖塊；擷取範圍稍大於圓形外圈。
    half = max(28, int(round(radius * 1.75)))
    x0 = max(0, int(round(cx)) - half)
    x1 = min(w, int(round(cx)) + half + 1)
    y0 = max(0, int(round(cy)) - half)
    y1 = min(h, int(round(cy)) + half + 1)

    patch = image_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None, 0.0, 0.0

    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    scores = []
    for symbol_id, template in templates.items():
        score = _best_template_score(patch_gray, template)
        scores.append((score, symbol_id))

    if not scores:
        return None, 0.0, 0.0

    scores.sort(reverse=True)
    best_score, best_id = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    margin = best_score - second_score

    if best_score < CLASSIFY_MIN_SCORE or margin < CLASSIFY_MIN_MARGIN:
        return None, best_score, margin

    return best_id, best_score, margin


def _classify_centers(image_bgr, centers, templates):
    counts = {i: 0 for i in range(13)}
    classified = []
    unknown = 0

    for center in centers:
        symbol_id, score, margin = _classify_one_center(
            image_bgr,
            center,
            templates
        )

        if symbol_id is None:
            unknown += 1
            classified.append({
                "center": center,
                "symbol_id": None,
                "score": score,
                "margin": margin
            })
        else:
            counts[symbol_id] += 1
            classified.append({
                "center": center,
                "symbol_id": symbol_id,
                "score": score,
                "margin": margin
            })

    return counts, unknown, classified


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
            method = "PDF 向量符號定位"
            note = "保留原本向量圓形路徑定位方式。"
        else:
            centers = _raster_socket_centers(image_bgr)
            method = "高解析度圖面圓形辨識"
            note = "保留原本 HoughCircles 圓形定位方式。"

        if not centers:
            raise ValueError(
                f"PDF 第 {page_number} 頁沒有找到可辨識的插座外圈，請確認頁碼。"
            )

        counts, unknown, classified = _classify_centers(
            image_bgr,
            centers,
            templates
        )

        return {
            "count": len(centers),
            "counts": counts,
            "unknown": unknown,
            "image": _jpeg_data_url(_mark_centers(image_bgr, centers)),
            "method": method,
            "note": note
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

    counts, unknown, classified = _classify_centers(
        image,
        centers,
        templates
    )

    return {
        "count": len(centers),
        "counts": counts,
        "unknown": unknown,
        "image": _jpeg_data_url(_mark_centers(image, centers)),
        "method": "高解析度圖面圓形辨識",
        "note": "保留原本 HoughCircles 圓形定位方式。"
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
            "分類依據為 static/symbols/0.png～12.png 模板，"
            "只對原本紅圈定位到的每個設備位置做局部符號比對。"
            "沒有讀取圖面右側圖例表、沒有使用 OCR，也沒有用像素差異代替設備辨識。"
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

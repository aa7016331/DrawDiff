
import os
import csv
import io
import base64
from pathlib import Path

import cv2
import fitz
import numpy as np
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
@media(max-width:850px){body{padding:12px}.upload-grid,.compare-grid{grid-template-columns:1fr}.main-result{grid-template-columns:1fr 1fr}.card{padding:16px}}
</style>
</head>
<body>
<header>
  <h1>DrawDiff｜插座增減辨識</h1>
  <div class="sub">上傳原始圖與設計師修改圖，自動清點電氣及弱電插座總數，計算增加與減少。</div>
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
    <thead><tr><th>設備項目</th><th>原圖數量</th><th>修改後數量</th><th>增加</th><th>減少</th><th>淨增減</th></tr></thead>
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
  statusEl.textContent='正在讀取指定頁面並辨識插座，請勿關閉頁面。';
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
    const tr=document.createElement('tr');
    const values=['全部插座（電氣＋弱電）',data.old_count,data.new_count,data.added_count,data.removed_count,(net>0?'+':'')+net];
    values.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td);});
    const body=document.getElementById('resultRows');body.replaceChildren(tr);
    latestRows=[{name:'全部插座（電氣＋弱電）',old_count:data.old_count,new_count:data.new_count,difference:net}];
    document.getElementById('resultNote').textContent='統計目標為平面圖中的電氣插座與弱電插座。紅圈為自動辨識位置，請核對是否都落在插座符號上；若發現誤圈或漏圈，請勿直接將結果作為請款依據。';
    exportButton.disabled=false;
    statusEl.textContent='辨識完成。請檢查左右圖面的紅圈是否都落在插座符號上。';
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


PLAN_X0 = 0.11
PLAN_X1 = 0.73
PLAN_Y0 = 0.20
PLAN_Y1 = 0.84
RENDER_SCALE = 4.0


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
    """在插座符號中心畫紅圈。"""
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
    CAD 向量 PDF：定位平面圖中重複的插座外圈圓形路徑。
    排除右側圖例表格、標題欄及頁面外框。
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

    from collections import Counter
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
    """掃描 PDF：在平面圖區域偵測插座外圈。"""
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

        rgb = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(pix.height, pix.width, pix.n)

        image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        vector_centers = _vector_socket_centers(page)

        if vector_centers:
            centers = vector_centers
            method = "PDF 向量符號定位"
            note = "此頁保留向量圖元，以重複的插座外圈路徑計數。"
        else:
            centers = _raster_socket_centers(image_bgr)
            method = "高解析度圖面圓形辨識"
            note = "此頁為影像圖面，請核對紅圈是否都落在插座上。"

        if not centers:
            raise ValueError(
                "這一頁沒有找到可辨識的插座符號。請確認頁碼。"
            )

        marked = _mark_centers(image_bgr, centers)

        return {
            "count": len(centers),
            "image": _jpeg_data_url(marked),
            "method": method,
            "note": note
        }
    finally:
        document.close()


def _analyze_image(file_bytes):
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("圖片無法讀取。")

    h, w = image.shape[:2]

    if max(h, w) < 1600:
        raise ValueError(
            "這張圖片解析度太低，無法可靠清點小型插座符號。請上傳原始 PDF。"
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
        return jsonify(
            error="請同時選擇原始圖 PDF 與設計師修改圖 PDF。"
        ), 400

    try:
        old_page = int(request.form.get("old_page", "4"))
        new_page = int(request.form.get("new_page", "3"))

        if old_page < 1 or new_page < 1:
            raise ValueError("PDF 頁碼必須從 1 開始。")

        old_result = analyze_upload(old_file, old_page)
        new_result = analyze_upload(new_file, new_page)

        old_count = old_result["count"]
        new_count = new_result["count"]
        difference = new_count - old_count

        added = max(difference, 0)
        removed = max(-difference, 0)

        return jsonify(
            old_count=old_count,
            new_count=new_count,
            difference=difference,
            added_count=added,
            removed_count=removed,
            old_image=old_result["image"],
            new_image=new_result["image"],
            old_method=old_result["method"],
            new_method=new_result["method"],
            old_note=old_result["note"],
            new_note=new_result["note"],
            rows=[{
                "name": "全部插座（電氣＋弱電）",
                "old_count": old_count,
                "new_count": new_count,
                "difference": difference
            }]
        )

    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    except Exception as exc:
        app.logger.exception("socket count failed")
        return jsonify(error=f"辨識失敗：{exc}"), 500


@app.post("/export")
def export_csv():
    data = request.get_json(silent=True) or {}
    rows = data.get("rows", [])

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "設備項目",
        "原圖數量",
        "修改後數量",
        "增加",
        "減少",
        "淨增減"
    ])

    for row in rows:
        diff = int(row.get("difference", 0))

        writer.writerow([
            row.get("name", "全部插座"),
            row.get("old_count", 0),
            row.get("new_count", 0),
            max(diff, 0),
            max(-diff, 0),
            diff
        ])

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                "attachment; filename=DrawDiff插座增減統計.csv"
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

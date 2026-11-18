import os
import csv
import io
import base64
import zipfile
from pathlib import Path
from collections import Counter

import cv2
import fitz
import numpy as np
from flask import Flask, render_template_string, request, jsonify, Response

BASE = Path(__file__).resolve().parent
SYMBOL_DIR = BASE / "static" / "symbols"
ALLOWED = {"pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# 13 張範本直接內嵌，避免新舊範本版本不一致。
_EMBEDDED_SYMBOLS_ZIP_B64 = "__EMBEDDED__"

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

CLASSIFY_CANVAS_SIZE = 176
CLASSIFY_EDGE_TOLERANCE = 2.5
CLASSIFY_SCALE_VARIANTS = (0.92, 1.0, 1.08)
CLASSIFY_CODE_MIN_MARGIN = 0.03
CLASSIFY_BODY_ONLY_MIN_SCORE = 0.86
CLASSIFY_BODY_ONLY_MIN_MARGIN = 0.10

GLYPH_BOXES = {
    2: (40, 17, 47, 29),
    4: (39, 14, 47, 27),
    5: (39, 16, 47, 28),
    6: (38, 17, 46, 28),
    7: (38, 18, 48, 29),
    9: (39, 15, 46, 28),
    10: (40, 30, 48, 43),
    11: (39, 38, 46, 47),
    12: (39, 17, 46, 28),
}

INTERNAL_GLYPH_BOXES = {
    8: (21, 8, 32, 25),
}

CODE_SYMBOL_IDS = (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)
BODY_ONLY_SYMBOL_IDS = (1, 3)

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
    symbol_id: group
    for group, ids in GLYPH_GROUPS.items()
    for symbol_id in ids
}

GLYPH_CODE_CONFLICT_MARGIN = {
    2: 0.035, 4: 0.010, 5: 0.070, 6: 0.030, 7: 0.025,
    8: 0.035, 9: 0.010, 10: 0.045, 11: 0.030, 12: 0.010,
}

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

INDEX_HTML = r'''
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DrawDiff｜13 種設備符號辨識</title>
<style>
:root{font-family:Arial,"Noto Sans TC",sans-serif;color:#152432;background:#f2f5f7}
*{box-sizing:border-box}
body{margin:0;padding:24px}
main,header{max-width:1420px;margin:0 auto}
header{margin-bottom:20px}
h1{font-size:25px;margin:0 0 6px}
.sub{color:#627381;font-size:14px}
.card{background:#fff;border:1px solid #d7e0e6;border-radius:14px;padding:22px;margin-bottom:18px}
h2{font-size:20px;margin:0 0 18px}
.upload-grid,.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.upload-box{border:1px dashed #aabac5;background:#fbfcfd;padding:16px;border-radius:10px}
.upload-box strong{display:block;margin-bottom:8px}
input[type=file]{display:block;width:100%;margin-top:8px}
.page-row{display:flex;align-items:center;gap:10px;margin-top:12px;color:#546675;font-size:14px}
input[type=number]{width:84px;padding:8px;border:1px solid #cbd6dd;border-radius:8px}
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
.image-panel{border:1px solid #d7e0e6;background:#fafcfd;border-radius:10px;padding:12px;min-width:0}
.image-title{text-align:center;font-weight:700;margin-bottom:8px}
.image-panel img{width:100%;height:auto;display:block;background:#fff;border:1px solid #e5eaee}
.method{font-size:12px;color:#667784;margin-top:8px;line-height:1.45}
table{border-collapse:collapse;width:100%;margin-top:20px}
th,td{border-bottom:1px solid #e3e9ed;padding:10px;text-align:center}
th{background:#edf2f5}
td:first-child,th:first-child{text-align:left}
.note{color:#687987;font-size:13px;margin-top:12px;line-height:1.5}
.symbol-cell{display:flex;align-items:center;gap:10px;text-align:left;min-width:260px}
.symbol-cell img{width:48px;height:44px;object-fit:contain;background:#fff;border:1px solid #e5eaee;border-radius:5px;padding:2px}
.positive{color:#13724e;font-weight:700}
.negative{color:#b42318;font-weight:700}
.total-row{font-weight:800;background:#f1f5f7}
@media(max-width:850px){body{padding:12px}.upload-grid,.compare-grid{grid-template-columns:1fr}.main-result{grid-template-columns:1fr 1fr}.card{padding:16px}}
</style>
</head>
<body>
<header>
<h1>DrawDiff｜13 種設備符號辨識</h1>
<div class="sub">原圖與修改圖左右比對，依 13 種設備符號逐項計數及增減。</div>
</header>
<main>
<section class="card">
<h2>上傳圖面</h2>
<form id="uploadForm">
<div class="upload-grid">
<div class="upload-box">
<strong>原始圖</strong>
<div class="sub">上傳原始 PDF，預設第 4 頁</div>
<input id="oldFile" name="old" type="file" accept=".pdf,application/pdf" required>
<div class="page-row">PDF 頁碼 <input name="old_page" type="number" min="1" value="4"></div>
</div>
<div class="upload-box">
<strong>設計師修改圖</strong>
<div class="sub">上傳修改 PDF，預設第 3 頁</div>
<input id="newFile" name="new" type="file" accept=".pdf,application/pdf" required>
<div class="page-row">PDF 頁碼 <input name="new_page" type="number" min="1" value="3"></div>
</div>
</div>
<div class="actions">
<button class="primary" id="analyzeButton" type="submit">開始辨識</button>
<button class="secondary" id="exportButton" type="button" disabled>匯出 CSV</button>
</div>
</form>
<div id="status"></div>
</section>
<section class="card" id="result">
<h2>設備增減結果</h2>
<div class="summary" id="summary"></div>
<div class="main-result">
<div class="metric"><div class="label">原圖紅圈定位數</div><div class="value" id="oldCount">—</div></div>
<div class="metric"><div class="label">修改圖紅圈定位數</div><div class="value" id="newCount">—</div></div>
<div class="metric"><div class="label">紅圈總數差</div><div class="value" id="netCount">—</div></div>
</div>
<div class="compare-grid">
<div class="image-panel">
<div class="image-title">原圖（紅圈為定位位置）</div>
<img id="oldResult" alt="原圖定位結果">
<div class="method" id="oldMethod"></div>
</div>
<div class="image-panel">
<div class="image-title">設計師修改圖（紅圈為定位位置）</div>
<img id="newResult" alt="修改圖定位結果">
<div class="method" id="newMethod"></div>
</div>
</div>
<table>
<thead><tr><th>編號／設備項目</th><th>原圖</th><th>修改後</th><th>增加</th><th>減少</th><th>淨增減</th></tr></thead>
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
form.addEventListener('submit',async(event)=>{
event.preventDefault();
const oldFile=document.getElementById('oldFile').files[0];
const newFile=document.getElementById('newFile').files[0];
if(!oldFile||!newFile){statusEl.textContent='請同時選擇原圖與修改圖。';statusEl.className='error';return;}
const payload=new FormData(form);
analyzeButton.disabled=true;exportButton.disabled=true;
statusEl.textContent='正在辨識圖面，請稍候。';statusEl.className='';
try{
const response=await fetch('/analyze',{method:'POST',body:payload});
const data=await response.json();
if(!response.ok)throw new Error(data.error||'辨識失敗');
document.getElementById('result').style.display='block';
document.getElementById('oldCount').textContent=data.old_detected_count;
document.getElementById('newCount').textContent=data.new_detected_count;
const net=data.difference;
document.getElementById('netCount').textContent=(net>0?'+':'')+net;
const summary=document.getElementById('summary');summary.replaceChildren();
[['原圖 '+data.old_detected_count+' 個',''],['修改後 '+data.new_detected_count+' 個',''],['增加 '+data.added_count+' 個','plus'],['減少 '+data.removed_count+' 個','minus']].forEach(([label,cls])=>{
const d=document.createElement('div');d.className='pill '+cls;d.textContent=label;summary.appendChild(d);
});
document.getElementById('oldResult').src=data.old_image;
document.getElementById('newResult').src=data.new_image;
document.getElementById('oldMethod').textContent=data.old_method+'。'+data.old_note;
document.getElementById('newMethod').textContent=data.new_method+'。'+data.new_note;
const body=document.getElementById('resultRows');body.replaceChildren();latestRows=data.rows||[];
latestRows.forEach(row=>{
const tr=document.createElement('tr');const item=document.createElement('td');item.className='symbol-cell';
const name=document.createElement('span');name.textContent=row.id+'｜'+row.name;item.appendChild(name);tr.appendChild(item);
[row.old_count,row.new_count,row.added_count,row.removed_count,(row.difference>0?'+':'')+row.difference].forEach((v,i)=>{
const td=document.createElement('td');td.textContent=v;
if(i===2&&row.added_count>0)td.className='positive';
if(i===3&&row.removed_count>0)td.className='negative';
tr.appendChild(td);
});
body.appendChild(tr);
});
const totalTr=document.createElement('tr');totalTr.className='total-row';
const totalLabel=document.createElement('td');totalLabel.textContent='合計（含未分類）';totalTr.appendChild(totalLabel);
const oldClass=data.old_classified_count+data.old_unknown_count;
const newClass=data.new_classified_count+data.new_unknown_count;
[oldClass,newClass,data.added_count,data.removed_count,(newClass-oldClass>0?'+':'')+(newClass-oldClass)].forEach(v=>{
const td=document.createElement('td');td.textContent=v;totalTr.appendChild(td);
});
body.appendChild(totalTr);
document.getElementById('resultNote').textContent=data.row_source+'；'+data.template_status;
exportButton.disabled=false;
statusEl.textContent='辨識完成。請核對未分類項目，不要直接將未分類數量當成設備增減。';
document.getElementById('result').scrollIntoView({behavior:'smooth',block:'start'});
}catch(err){statusEl.textContent=err.message;statusEl.className='error';}
finally{analyzeButton.disabled=false;}
});
exportButton.addEventListener('click',async()=>{
if(!latestRows.length)return;
try{
const r=await fetch('/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rows:latestRows})});
if(!r.ok)throw new Error('CSV 匯出失敗');
const blob=await r.blob();const url=URL.createObjectURL(blob);
const a=document.createElement('a');a.href=url;a.download='DrawDiff_13種設備增減統計.csv';document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);
}catch(err){statusEl.textContent=err.message;statusEl.className='error';}
});
</script>
</body>
</html>
'''

def _filename_ext(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

def _jpeg_data_url(image_bgr, max_side=1900, quality=84):
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("影像輸出失敗。")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

# 以下三個函式維持原本確認過的版本，不更動。
def _mark_centers(image_bgr, centers):
    marked = image_bgr.copy()
    for cx, cy, radius in centers:
        cv2.circle(marked, (int(round(cx)), int(round(cy))), int(round(radius)), (0, 0, 255), 2, cv2.LINE_AA)
    return marked

def _vector_socket_centers(page):
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
    selected = [item for item in candidates if abs(item[2] - dominant_size) <= 0.25]
    selected.sort(key=lambda item: item[2], reverse=True)
    kept = []
    for cx, cy, diameter in selected:
        if any((cx-x)**2+(cy-y)**2 < 2.0**2 for x,y,_ in kept):
            continue
        kept.append((cx, cy, diameter))
    return [(cx*RENDER_SCALE, cy*RENDER_SCALE, diameter*RENDER_SCALE*0.58) for cx,cy,diameter in kept]

def _raster_socket_centers(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x0 = int(w * PLAN_X0); x1 = int(w * PLAN_X1)
    y0 = int(h * PLAN_Y0); y1 = int(h * PLAN_Y1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    small = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (5,5), 1.2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=27,
        param1=80, param2=39, minRadius=10, maxRadius=19)
    if circles is None:
        return []
    results=[]
    for cx,cy,radius in np.round(circles[0]).astype(int):
        results.append((float(x0+cx*2),float(y0+cy*2),float(max(radius*2,12))))
    return results

def _clean_template(image):
    if image is None:
        return None
    gray=image.copy()
    if gray.ndim==3:
        gray=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,225,255,cv2.THRESH_BINARY_INV)
    coords=cv2.findNonZero(binary)
    if coords is None:return None
    x,y,w,h=cv2.boundingRect(coords)
    if w<5 or h<5:return None
    return gray[y:y+h,x:x+w]

def load_symbol_templates():
    templates={};missing=[]
    try:
        raw_zip=base64.b64decode(_EMBEDDED_SYMBOLS_ZIP_B64)
        with zipfile.ZipFile(io.BytesIO(raw_zip),"r") as archive:
            for symbol_id in range(13):
                raw=archive.read(f"{symbol_id}.png")
                image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
                if image is None or _clean_template(image) is None:
                    missing.append(f"{symbol_id}.png")
                else:
                    templates[symbol_id]=image
    except Exception as exc:
        raise ValueError(f"內嵌符號範本載入失敗：{exc}")
    return templates,missing

def _edge_image(gray):
    blur=cv2.GaussianBlur(gray,(3,3),0)
    return cv2.Canny(blur,40,120)

def _extract_center_patch(image_bgr,center,canvas_size):
    cx,cy,_=center;half=canvas_size//2
    center_x=int(round(cx));center_y=int(round(cy))
    x0=center_x-half;y0=center_y-half;x1=x0+canvas_size;y1=y0+canvas_size
    patch=np.full((canvas_size,canvas_size),255,dtype=np.uint8)
    sx0=max(0,x0);sy0=max(0,y0);sx1=min(image_bgr.shape[1],x1);sy1=min(image_bgr.shape[0],y1)
    if sx1<=sx0 or sy1<=sy0:return patch
    source=cv2.cvtColor(image_bgr[sy0:sy1,sx0:sx1],cv2.COLOR_BGR2GRAY)
    dx=sx0-x0;dy=sy0-y0
    patch[dy:dy+source.shape[0],dx:dx+source.shape[1]]=source
    return patch

def _masked_edge_f1(template_edges,target_edges,mask,tolerance=CLASSIFY_EDGE_TOLERANCE):
    tp=(template_edges>0)&mask;pp=(target_edges>0)&mask
    tc=int(np.count_nonzero(tp));pc=int(np.count_nonzero(pp))
    if tc<4 or pc<4:return 0.0
    d1=cv2.distanceTransform((~pp).astype(np.uint8),cv2.DIST_L2,3)
    d2=cv2.distanceTransform((~tp).astype(np.uint8),cv2.DIST_L2,3)
    precision=float(np.mean(d1[tp]<=tolerance))
    recall=float(np.mean(d2[pp]<=tolerance))
    return float(2*precision*recall/(precision+recall+1e-6))

def _masked_ink_f1(template_gray,target_gray,mask,threshold=190):
    ti=(template_gray<threshold)&mask;pi=(target_gray<threshold)&mask
    tc=int(np.count_nonzero(ti));pc=int(np.count_nonzero(pi))
    if tc<4 or pc<4:return 0.0
    inter=int(np.count_nonzero(ti&pi))
    return float(2*inter/(tc+pc+1e-6))

def _normalize_glyph(gray):
    if gray is None or gray.size==0:return np.zeros((64,64),dtype=np.uint8)
    binary=(gray<190).astype(np.uint8)*255
    ys,xs=np.where(binary>0)
    if len(xs)==0:return np.zeros((64,64),dtype=np.uint8)
    binary=binary[ys.min():ys.max()+1,xs.min():xs.max()+1]
    height,width=binary.shape
    scale=min(52.0/max(width,1),52.0/max(height,1))
    nw=max(1,int(round(width*scale)));nh=max(1,int(round(height*scale)))
    resized=cv2.resize(binary,(nw,nh),interpolation=cv2.INTER_NEAREST)
    canvas=np.zeros((64,64),dtype=np.uint8)
    x0=(64-nw)//2;y0=(64-nh)//2
    canvas[y0:y0+nh,x0:x0+nw]=resized
    return canvas

def _hog_descriptor():
    return cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)

_HOG=None
_GLYPH_HOG_TEMPLATES=None

def _get_glyph_hog_templates(templates):
    global _HOG,_GLYPH_HOG_TEMPLATES
    if _GLYPH_HOG_TEMPLATES is not None:return _GLYPH_HOG_TEMPLATES
    _HOG=_hog_descriptor();results={}
    boxes=dict(GLYPH_BOXES);boxes.update(INTERNAL_GLYPH_BOXES)
    for symbol_id,box in boxes.items():
        image=templates.get(symbol_id)
        if image is None:continue
        x0,y0,x1,y1=box
        glyph=_normalize_glyph(image[y0:y1,x0:x1])
        vector=_HOG.compute(glyph).flatten().astype(np.float32)
        norm=float(np.linalg.norm(vector))
        if norm>1e-8:vector/=norm
        results[symbol_id]=vector
    _GLYPH_HOG_TEMPLATES=results
    return results

def _template_features(template_gray,symbol_id,target_radius,scale_variant):
    if symbol_id not in TEMPLATE_CIRCLE_GEOMETRY:return None
    source=template_gray[:,:min(template_gray.shape[1],60)]
    ax,ay,tr=TEMPLATE_CIRCLE_GEOMETRY[symbol_id]
    scale=float(target_radius)*float(scale_variant)/float(tr)
    center=CLASSIFY_CANVAS_SIZE//2
    matrix=cv2.getRotationMatrix2D((ax,ay),0.0,scale)
    matrix[0,2]+=center-ax;matrix[1,2]+=center-ay
    warped=cv2.warpAffine(source,matrix,(CLASSIFY_CANVAS_SIZE,CLASSIFY_CANVAS_SIZE),
        flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
    edges=_edge_image(warped)
    yy,xx=np.ogrid[:source.shape[0],:source.shape[1]]
    radial=np.sqrt((xx-ax)**2+(yy-ay)**2)
    body_source=(radial<tr*0.72).astype(np.uint8)*255
    body_mask=cv2.warpAffine(body_source,matrix,(CLASSIFY_CANVAS_SIZE,CLASSIFY_CANVAS_SIZE),
        flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
    glyph_box=INTERNAL_GLYPH_BOXES.get(symbol_id,GLYPH_BOXES.get(symbol_id))
    glyph_mask=None;glyph_roi=None
    if glyph_box:
        gx0,gy0,gx1,gy1=glyph_box
        glyph_source=np.zeros(source.shape,dtype=np.uint8)
        glyph_source[max(0,gy0-1):min(source.shape[0],gy1+1),
            max(0,gx0-1):min(source.shape[1],gx1+1)]=255
        glyph_mask=cv2.warpAffine(glyph_source,matrix,(CLASSIFY_CANVAS_SIZE,CLASSIFY_CANVAS_SIZE),
            flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
        xa=center+(gx0-ax)*scale;xb=center+(gx1-ax)*scale
        ya=center+(gy0-ay)*scale;yb=center+(gy1-ay)*scale
        px=max(2,int(round((xb-xa)*0.22)));py=max(2,int(round((yb-ya)*0.22)))
        glyph_roi=(max(0,int(round(xa-px))),max(0,int(round(ya-py))),
            min(CLASSIFY_CANVAS_SIZE,int(round(xb+px))),
            min(CLASSIFY_CANVAS_SIZE,int(round(yb+py))))
    return edges,body_mask,glyph_mask,glyph_roi,warped

def _score_glyph_hog(patch,glyph_roi,symbol_id,hog_templates):
    if glyph_roi is None or symbol_id not in hog_templates:return 0.0
    x0,y0,x1,y1=glyph_roi
    if x1<=x0 or y1<=y0:return 0.0
    normalized=_normalize_glyph(patch[y0:y1,x0:x1])
    vector=_HOG.compute(normalized).flatten().astype(np.float32)
    norm=float(np.linalg.norm(vector))
    if norm<=1e-8:return 0.0
    vector/=norm
    return float(np.clip(np.dot(vector,hog_templates[symbol_id]),0.0,1.0))

def _score_glyph_group_evidence(patch,glyph_roi,hog_templates,symbol_id):
    if glyph_roi is None or symbol_id not in SYMBOL_GLYPH_GROUP:return {},None,0.0,0.0
    x0,y0,x1,y1=glyph_roi
    if x1<=x0 or y1<=y0:return {},None,0.0,0.0
    normalized=_normalize_glyph(patch[y0:y1,x0:x1])
    vector=_HOG.compute(normalized).flatten().astype(np.float32)
    norm=float(np.linalg.norm(vector))
    if norm<=1e-8:return {},None,0.0,0.0
    vector/=norm
    scores={}
    for group,ids in GLYPH_GROUPS.items():
        values=[float(np.dot(vector,hog_templates[sid])) for sid in ids if sid in hog_templates]
        if values:scores[group]=float(np.clip(max(values),0.0,1.0))
    expected=SYMBOL_GLYPH_GROUP[symbol_id]
    expected_score=scores.get(expected,0.0)
    best_other=max((v for k,v in scores.items() if k!=expected),default=0.0)
    conflict=float(best_other-expected_score)
    best_group=max(scores,key=scores.get) if scores else None
    return scores,best_group,expected_score,conflict

def _score_one_candidate(patch,patch_edges,templates,symbol_id,target_radius,hog_templates):
    best=None
    for scale_variant in CLASSIFY_SCALE_VARIANTS:
        feature=_template_features(templates[symbol_id],symbol_id,target_radius,scale_variant)
        if feature is None:continue
        template_edges,body_mask,glyph_mask,glyph_roi,template_gray=feature
        body_score=_masked_edge_f1(template_edges,patch_edges,body_mask)
        ink_score=_masked_ink_f1(template_gray,patch,body_mask)
        if glyph_mask is None:
            glyph_score=0.0;hog_score=0.0;combined=body_score
            group_scores={};group_best=None;group_expected=0.0;group_conflict=0.0
        else:
            glyph_score=_masked_edge_f1(template_edges,patch_edges,glyph_mask)
            hog_score=_score_glyph_hog(patch,glyph_roi,symbol_id,hog_templates)
            group_scores,group_best,group_expected,group_conflict=_score_glyph_group_evidence(
                patch,glyph_roi,hog_templates,symbol_id)
            combined=0.15*body_score+0.40*glyph_score+0.45*hog_score
        row={"score":float(combined),"body":float(body_score),"ink":float(ink_score),
            "glyph":float(glyph_score),"hog":float(hog_score),
            "glyph_group_scores":group_scores,"glyph_group_best":group_best,
            "glyph_group_expected":SYMBOL_GLYPH_GROUP.get(symbol_id),
            "glyph_group_expected_score":float(group_expected),
            "glyph_group_conflict":float(group_conflict),"scale":float(scale_variant)}
        if best is None or row["score"]>best["score"]:best=row
    return best or {"score":0.0,"body":0.0,"ink":0.0,"glyph":0.0,"hog":0.0,
        "glyph_group_scores":{},"glyph_group_best":None,
        "glyph_group_expected":SYMBOL_GLYPH_GROUP.get(symbol_id),
        "glyph_group_expected_score":0.0,"glyph_group_conflict":0.0,"scale":1.0}

def _get_center_template_scores(image_bgr,center,templates,target_radius,hog_templates):
    patch=_extract_center_patch(image_bgr,center,CLASSIFY_CANVAS_SIZE)
    score_map={}
    for angle,k in ((0,0),(90,1),(180,2),(270,3)):
        oriented=patch if k==0 else np.ascontiguousarray(np.rot90(patch,k))
        patch_edges=_edge_image(oriented)
        for symbol_id in range(1,13):
            if symbol_id not in templates:continue
            item=_score_one_candidate(oriented,patch_edges,templates,symbol_id,target_radius,hog_templates)
            item=dict(item);item["orientation"]=angle
            prev=score_map.get(symbol_id)
            if prev is None or item["score"]>prev["score"]:score_map[symbol_id]=item
    return score_map

def _decide_symbol(score_map):
    thresholds={
        2:{"score":0.70,"glyph":0.64,"hog":0.72,"body":0.55,"ink":0.00},
        4:{"score":0.70,"glyph":0.73,"hog":0.60,"body":0.55,"ink":0.42},
        5:{"score":0.70,"glyph":0.64,"hog":0.74,"body":0.55,"ink":0.00},
        6:{"score":0.70,"glyph":0.56,"hog":0.70,"body":0.55,"ink":0.60},
        7:{"score":0.70,"glyph":0.64,"hog":0.72,"body":0.55,"ink":0.00},
        8:{"score":0.73,"glyph":0.72,"hog":0.68,"body":0.58,"ink":0.00},
        9:{"score":0.70,"glyph":0.73,"hog":0.60,"body":0.55,"ink":0.60},
        10:{"score":0.70,"glyph":0.64,"hog":0.74,"body":0.55,"ink":0.48},
        11:{"score":0.70,"glyph":0.64,"hog":0.75,"body":0.52,"ink":0.40},
        12:{"score":0.70,"glyph":0.73,"hog":0.60,"body":0.55,"ink":0.35},
    }
    eligible=[];reliable=[]
    for sid in CODE_SYMBOL_IDS:
        item=score_map.get(sid)
        if not item:continue
        th=thresholds[sid]
        conflict=float(item.get("glyph_group_conflict",0.0))
        allowed=GLYPH_CODE_CONFLICT_MARGIN.get(sid,0.035)
        consistent=conflict<=allowed
        passes=(item.get("score",0)>=th["score"] and item.get("glyph",0)>=th["glyph"]
            and item.get("hog",0)>=th["hog"] and item.get("body",0)>=th["body"]
            and item.get("ink",0)>=th["ink"] and consistent)
        if passes:
            eligible.append((item["score"],sid,item))
            reliable.append(sid)
    if eligible:
        eligible.sort(key=lambda x:x[0],reverse=True)
        best_score,best_id,best_item=eligible[0]
        second=eligible[1][0] if len(eligible)>1 else 0.0
        margin=best_score-second
        if margin>=CLASSIFY_CODE_MIN_MARGIN:
            j=score_map.get(8)
            if best_id==7 and j and j.get("glyph",0)>=0.60 and j.get("hog",0)>=0.55 and j.get("body",0)>=0.45:
                return None,best_score,margin,best_item,"possible_internal_J"
            return best_id,best_score,margin,best_item,"glyph_visual_rotated"
        return None,best_score,margin,best_item,"ambiguous_code"
    if reliable:
        top=max((score_map[sid] for sid in reliable),key=lambda x:x.get("score",0))
        return None,top["score"],0.0,top,"code_needs_review"
    ranked=sorted(((score_map[sid]["body"],sid,score_map[sid])
        for sid in BODY_ONLY_SYMBOL_IDS if sid in score_map),reverse=True,key=lambda x:x[0])
    if not ranked:return None,0.0,0.0,{},"no_match"
    score,sid,item=ranked[0]
    second=ranked[1][0] if len(ranked)>1 else 0.0
    margin=score-second
    if score>=CLASSIFY_BODY_ONLY_MIN_SCORE and margin>=CLASSIFY_BODY_ONLY_MIN_MARGIN:
        return sid,score,margin,item,"body_visual_rotated"
    return None,score,margin,item,"needs_review"

def _classify_one_center(image_bgr,center,templates,radius_scale=1.0,hog_templates=None):
    radius=float(center[2])*float(radius_scale)
    radius=float(np.clip(radius,20.0,36.0))
    if hog_templates is None:hog_templates=_get_glyph_hog_templates(templates)
    scores=_get_center_template_scores(image_bgr,center,templates,radius,hog_templates)
    return _decide_symbol(scores)

def _classify_centers(image_bgr,centers,templates,radius_scale=1.0):
    counts={i:0 for i in range(13)}
    classified=[];unknown=0
    hog_templates=_get_glyph_hog_templates(templates)
    for center in centers:
        sid,score,margin,detail,decision=_classify_one_center(
            image_bgr,center,templates,radius_scale,hog_templates)
        if sid is None:unknown+=1
        else:counts[sid]+=1
        classified.append({"center":center,"symbol_id":sid,"score":float(score),
            "margin":float(margin),"decision":decision,"features":detail})
    glyph_matched=sum(1 for x in classified if x["symbol_id"] not in (None,1,3))
    return counts,unknown,classified,glyph_matched

def _analyze_pdf(file_bytes,page_number,templates):
    try:
        document=fitz.open(stream=file_bytes,filetype="pdf")
    except Exception as exc:
        raise ValueError(f"PDF 無法開啟：{exc}")
    try:
        index=page_number-1
        if index<0 or index>=len(document):
            raise ValueError(f"PDF 只有 {len(document)} 頁，指定的第 {page_number} 頁不存在。")
        page=document.load_page(index)
        pix=page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE,RENDER_SCALE),
            alpha=False,colorspace=fitz.csRGB)
        rgb=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)
        image_bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        vector_centers=_vector_socket_centers(page)
        if vector_centers:
            centers=vector_centers;radius_scale=1.0/1.16
            method="PDF 向量符號定位"
            note="保留原本向量圓形路徑定位方式；分類以圓心固定對位。"
        else:
            centers=_raster_socket_centers(image_bgr);radius_scale=1.0
            method="高解析度圖面圓形辨識"
            note="保留原本 HoughCircles 圓形定位方式；分類以圓心固定對位。"
        if not centers:
            raise ValueError(f"PDF 第 {page_number} 頁沒有找到可辨識的插座外圈，請確認頁碼。")
        counts,unknown,classified,glyph_matched=_classify_centers(
            image_bgr,centers,templates,radius_scale)
        return {"count":len(centers),"counts":counts,"unknown":unknown,
            "glyph_matched":glyph_matched,
            "image":_jpeg_data_url(_mark_centers(image_bgr,centers)),
            "method":method,
            "note":note+"；分類比對圖示內部結構及英文代碼，校正 0°／90°／180°／270°方向，不使用 OCR。"}
    finally:
        document.close()

def _analyze_image(file_bytes,templates):
    array=np.frombuffer(file_bytes,dtype=np.uint8)
    image=cv2.imdecode(array,cv2.IMREAD_COLOR)
    if image is None:raise ValueError("圖片無法讀取。")
    h,w=image.shape[:2]
    if max(h,w)<1600:
        raise ValueError("圖片解析度太低，無法可靠清點小型設備符號。請上傳原始 PDF。")
    centers=_raster_socket_centers(image)
    if not centers:raise ValueError("圖片中沒有找到可辨識的插座外圈。")
    counts,unknown,classified,glyph_matched=_classify_centers(image,centers,templates,1.0)
    return {"count":len(centers),"counts":counts,"unknown":unknown,
        "glyph_matched":glyph_matched,
        "image":_jpeg_data_url(_mark_centers(image,centers)),
        "method":"高解析度圖面圓形辨識",
        "note":"保留原本 HoughCircles 定位方式；分類使用圖示及英文代碼影像特徵，不使用 OCR。"}

def analyze_upload(upload,page_number,templates):
    filename=upload.filename or ""
    ext=filename.rsplit(".",1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED:raise ValueError("只支援 PDF、PNG、JPG、JPEG、BMP、TIF、TIFF。")
    raw=upload.read()
    if not raw:raise ValueError(f"檔案是空的：{filename}")
    if ext=="pdf":return _analyze_pdf(raw,page_number,templates)
    return _analyze_image(raw,templates)

def _build_rows(old_result,new_result):
    rows=[]
    for sid in range(13):
        old=int(old_result["counts"].get(sid,0))
        new=int(new_result["counts"].get(sid,0))
        diff=new-old
        rows.append({"id":sid,"name":SYMBOL_NAMES[sid],"old_count":old,
            "new_count":new,"added_count":max(diff,0),
            "removed_count":max(-diff,0),"difference":diff})
    old_u=int(old_result["unknown"]);new_u=int(new_result["unknown"])
    diff=new_u-old_u
    if old_u or new_u:
        rows.append({"id":"unknown","name":"未分類（需人工核對）",
            "old_count":old_u,"new_count":new_u,
            "added_count":max(diff,0),"removed_count":max(-diff,0),
            "difference":diff})
    return rows

@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

@app.post("/analyze")
def analyze_api():
    old_file=request.files.get("old");new_file=request.files.get("new")
    if not old_file or not new_file:
        return jsonify(error="請同時選擇原圖與設計師修改圖。"),400
    try:
        old_page=int(request.form.get("old_page","4"))
        new_page=int(request.form.get("new_page","3"))
        if old_page<1 or new_page<1:raise ValueError("PDF 頁碼必須從 1 開始。")
        templates,missing=load_symbol_templates()
        if missing:
            raise ValueError("內嵌範本不完整："+ "、".join(missing))
        old=analyze_upload(old_file,old_page,templates)
        new=analyze_upload(new_file,new_page,templates)
        rows=_build_rows(old,new)
        old_total=old["count"];new_total=new["count"]
        diff=new_total-old_total
        old_classified=sum(old["counts"].values())
        new_classified=sum(new["counts"].values())
        note=("分類以紅圈中心固定對位；圖示結構與英文代碼形狀共同比對，並校正四個正交方向。"
            "不使用 OCR、不讀取右側圖例／數量表，也不以整頁像素差異代替設備辨識。"
            "信心不足的候選保留未分類，不強制猜測。"
            "0 號為長方形符號，不在原本圓形紅圈定位器範圍內，尚未自動計入。")
        if old["unknown"] or new["unknown"]:
            note+=f"未分類：原圖 {old['unknown']} 個，修改圖 {new['unknown']} 個，需人工核對。"
        return jsonify(
            old_detected_count=old_total,new_detected_count=new_total,
            difference=diff,added_count=max(diff,0),removed_count=max(-diff,0),
            old_classified_count=old_classified,new_classified_count=new_classified,
            old_unknown_count=old["unknown"],new_unknown_count=new["unknown"],
            old_image=old["image"],new_image=new["image"],
            old_method=old["method"],new_method=new["method"],
            old_note=old["note"],new_note=new["note"],
            row_source=note,template_status="內嵌符號範本 13/13 已載入",
            rows=rows)
    except ValueError as exc:
        return jsonify(error=str(exc)),400
    except Exception as exc:
        app.logger.exception("DrawDiff analysis failed")
        return jsonify(error=f"辨識失敗：{exc}"),500

@app.post("/export")
def export_csv():
    data=request.get_json(silent=True) or {}
    rows=data.get("rows",[])
    output=io.StringIO()
    writer=csv.writer(output)
    writer.writerow(["設備編號","設備項目","原圖數量","修改後數量","增加","減少","淨增減"])
    old_total=new_total=added_total=removed_total=0
    for row in rows:
        old=int(row.get("old_count",0));new=int(row.get("new_count",0))
        diff=new-old
        old_total+=old;new_total+=new
        added_total+=max(diff,0);removed_total+=max(-diff,0)
        writer.writerow([row.get("id",""),row.get("name",""),old,new,
            max(diff,0),max(-diff,0),diff])
    writer.writerow(["","合計（含未分類）",old_total,new_total,
        added_total,removed_total,new_total-old_total])
    return Response("﻿"+output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition":
            "attachment; filename=DrawDiff_13種設備增減統計.csv"})

if __name__=="__main__":
    port=int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)

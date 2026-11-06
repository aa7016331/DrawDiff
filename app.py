
import os, io, uuid, tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import fitz

BASE = Path(__file__).resolve().parent
SYMBOL_DIR = BASE / "static" / "symbols"
ALLOWED = {"png","jpg","jpeg","pdf"}

LEGENDS = [
    "住戶開關箱與弱電箱整合",
    "馬桶專用插座 H:30cm",
    "抽油煙機插座 H:天花板內",
    "雙聯插座 H:30cm／電視牆 H:80cm",
    "雙聯插座 H:110cm",
    "洗碗機專用插座 H:45cm",
    "微波烤箱專用插座 220V",
    "洗衣機用插座（防雨型）H:120cm",
    "廚房橫櫃燈用電源出線口 H:170cm",
    "暖風乾燥機電源出口 220V",
    "雙聯紅色緊急插座（弱電箱用）",
    "雙聯紅色緊急插座（冰箱／電視牆）",
    "冷氣電源出口 220V",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

def allowed(name):
    return "." in name and name.rsplit(".",1)[1].lower() in ALLOWED

def read_upload(f):
    ext = f.filename.rsplit(".",1)[1].lower()
    raw = f.read()
    if ext == "pdf":
        doc = fitz.open(stream=raw, filetype="pdf")
        # First page is used; construction plans should be exported as one page per comparison.
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0,2.0), alpha=False)
        arr = np.frombuffer(pix.tobytes("png"), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        doc.close()
        return img
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Keep linework while reducing uneven background.
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

def nms(boxes, min_dist):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    for b in boxes:
        cx, cy = b[0] + b[2]/2, b[1] + b[3]/2
        if all(((cx-(k[0]+k[2]/2))**2 + (cy-(k[1]+k[3]/2))**2)**0.5 >= min_dist for k in keep):
            keep.append(b)
    return keep

def count_template(plan_gray, template_path):
    tmpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        return 0

    # Auto-scale search. Conservative threshold reduces double-counting.
    hits = []
    h0, w0 = tmpl.shape[:2]
    for scale in (0.55,0.65,0.75,0.85,1.0,1.15,1.3,1.5,1.75):
        w, h = max(8,int(w0*scale)), max(8,int(h0*scale))
        if w >= plan_gray.shape[1] or h >= plan_gray.shape[0]:
            continue
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        t = cv2.resize(tmpl,(w,h),interpolation=interp)
        result = cv2.matchTemplate(plan_gray,t,cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= 0.82)
        for y,x in zip(ys,xs):
            hits.append((int(x),int(y),w,h,float(result[y,x])))

    if not hits:
        return 0
    distance = max(12, min(w0,h0)*0.55)
    return len(nms(hits,distance))

def analyze(img):
    gray = preprocess(img)
    out = []
    for i, name in enumerate(LEGENDS):
        path = SYMBOL_DIR / f"{i}.png"
        out.append(count_template(gray,path))
    return out

@app.get("/")
def index():
    return render_template("index.html", legends=LEGENDS)

@app.post("/api/analyze")
def api_analyze():
    old = request.files.get("old")
    new = request.files.get("new")
    if not old or not new:
        return jsonify(error="請同時上傳原始圖與修改後圖"), 400
    if not allowed(old.filename) or not allowed(new.filename):
        return jsonify(error="僅支援 PNG、JPG、JPEG、PDF"), 400

    old_img = read_upload(old)
    new_img = read_upload(new)
    if old_img is None or new_img is None:
        return jsonify(error="有圖片無法讀取"), 400

    old_counts = analyze(old_img)
    new_counts = analyze(new_img)
    rows = []
    for i,name in enumerate(LEGENDS):
        rows.append({
            "name": name,
            "old": int(old_counts[i]),
            "new": int(new_counts[i]),
            "diff": int(new_counts[i]-old_counts[i]),
        })
    return jsonify(rows=rows)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

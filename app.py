import os
import csv
import io
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import fitz

BASE = Path(__file__).resolve().parent
SYMBOL_DIR = BASE / "static" / "symbols"

ALLOWED = {
    "png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"
}

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


def allowed(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED
    )


def read_upload(file):
    ext = file.filename.rsplit(".", 1)[1].lower()
    raw = file.read()

    if ext == "pdf":
        doc = fitz.open(stream=raw, filetype="pdf")

        page = doc.load_page(0)

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2.0, 2.0),
            alpha=False
        )

        arr = np.frombuffer(
            pix.tobytes("png"),
            np.uint8
        )

        image = cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR
        )

        doc.close()

        return image

    arr = np.frombuffer(raw, np.uint8)

    return cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )


def preprocess(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


def nms(boxes, min_dist):
    boxes = sorted(
        boxes,
        key=lambda x: x[4],
        reverse=True
    )

    keep = []

    for box in boxes:

        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2

        far_enough = all(
            (
                (
                    cx - (k[0] + k[2] / 2)
                ) ** 2
                +
                (
                    cy - (k[1] + k[3] / 2)
                ) ** 2
            ) ** 0.5 >= min_dist

            for k in keep
        )

        if far_enough:
            keep.append(box)

    return keep


def count_template(plan_gray, template_path):

    template = cv2.imread(
        str(template_path),
        cv2.IMREAD_GRAYSCALE
    )

    if template is None:
        return 0

    hits = []

    h0, w0 = template.shape[:2]

    scales = (
        0.55,
        0.65,
        0.75,
        0.85,
        1.0,
        1.15,
        1.3,
        1.5,
        1.75
    )

    for scale in scales:

        w = max(8, int(w0 * scale))
        h = max(8, int(h0 * scale))

        if (
            w >= plan_gray.shape[1]
            or h >= plan_gray.shape[0]
        ):
            continue

        interpolation = (
            cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_CUBIC
        )

        resized = cv2.resize(
            template,
            (w, h),
            interpolation=interpolation
        )

        result = cv2.matchTemplate(
            plan_gray,
            resized,
            cv2.TM_CCOEFF_NORMED
        )

        ys, xs = np.where(result >= 0.82)

        for y, x in zip(ys, xs):

            hits.append(
                (
                    int(x),
                    int(y),
                    w,
                    h,
                    float(result[y, x])
                )
            )

    if not hits:
        return 0

    distance = max(
        12,
        min(w0, h0) * 0.55
    )

    return len(
        nms(hits, distance)
    )


def analyze(image):

    gray = preprocess(image)

    counts = []

    for i in range(len(LEGENDS)):

        symbol_path = (
            SYMBOL_DIR / f"{i}.png"
        )

        counts.append(
            count_template(
                gray,
                symbol_path
            )
        )

    return counts


@app.get("/")
def index():

    return render_template(
        "index.html",
        items=LEGENDS
    )


@app.post("/analyze")
def analyze_api():

    old = request.files.get("old")
    new = request.files.get("new")

    if not old or not new:

        return jsonify(
            error="請同時上傳原始圖與修改後圖"
        ), 400

    if (
        not allowed(old.filename)
        or not allowed(new.filename)
    ):

        return jsonify(
            error="僅支援 PNG、JPG、JPEG、BMP、TIF、TIFF、PDF"
        ), 400

    old_image = read_upload(old)
    new_image = read_upload(new)

    if old_image is None or new_image is None:

        return jsonify(
            error="有圖片無法讀取"
        ), 400

    old_counts = analyze(old_image)
    new_counts = analyze(new_image)

    rows = []

    for i, name in enumerate(LEGENDS):

        rows.append(
            {
                "id": i,
                "name": name,
                "old": int(old_counts[i]),
                "new": int(new_counts[i]),
                "diff": int(
                    new_counts[i] - old_counts[i]
                ),
            }
        )

    return jsonify(rows=rows)


@app.post("/export")
def export_csv():

    data = request.get_json()
    rows = data.get("rows", [])

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow(
        [
            "設備／功能",
            "原圖數量",
            "新圖數量",
            "增減"
        ]
    )

    for row in rows:

        writer.writerow(
            [
                row.get("name", ""),
                row.get("old", 0),
                row.get("new", 0),
                row.get("diff", 0),
            ]
        )

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                "attachment; filename=DrawDiff.csv"
        }
    )


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )

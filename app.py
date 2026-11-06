import os
import csv
import io
import base64

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import fitz


ALLOWED = {
    "png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"
}

LEGENDS = [
    "圖面差異",
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

        doc = fitz.open(
            stream=raw,
            filetype="pdf"
        )

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


def encode_image(image):

    success, buffer = cv2.imencode(
        ".png",
        image
    )

    if not success:
        return None

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return "data:image/png;base64," + encoded


def align_images(old_image, new_image):

    old_gray = cv2.cvtColor(
        old_image,
        cv2.COLOR_BGR2GRAY
    )

    new_gray = cv2.cvtColor(
        new_image,
        cv2.COLOR_BGR2GRAY
    )

    old_h, old_w = old_gray.shape

    new_image = cv2.resize(
        new_image,
        (old_w, old_h),
        interpolation=cv2.INTER_AREA
    )

    new_gray = cv2.cvtColor(
        new_image,
        cv2.COLOR_BGR2GRAY
    )

    try:

        warp_matrix = np.eye(
            2,
            3,
            dtype=np.float32
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            100,
            1e-6
        )

        _, warp_matrix = cv2.findTransformECC(
            old_gray.astype(np.float32) / 255.0,
            new_gray.astype(np.float32) / 255.0,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria
        )

        aligned_new = cv2.warpAffine(
            new_image,
            warp_matrix,
            (old_w, old_h),
            flags=(
                cv2.INTER_LINEAR
                + cv2.WARP_INVERSE_MAP
            ),
            borderMode=cv2.BORDER_REPLICATE
        )

        return old_image, aligned_new

    except cv2.error:

        return old_image, new_image


def find_differences(old_image, new_image):

    old_image, new_image = align_images(
        old_image,
        new_image
    )

    old_gray = cv2.cvtColor(
        old_image,
        cv2.COLOR_BGR2GRAY
    )

    new_gray = cv2.cvtColor(
        new_image,
        cv2.COLOR_BGR2GRAY
    )

    old_gray = cv2.GaussianBlur(
        old_gray,
        (3, 3),
        0
    )

    new_gray = cv2.GaussianBlur(
        new_gray,
        (3, 3),
        0
    )

    diff = cv2.absdiff(
        old_gray,
        new_gray
    )

    _, thresh = cv2.threshold(
        diff,
        35,
        255,
        cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    image_area = (
        old_image.shape[0]
        * old_image.shape[1]
    )

    min_area = max(
        80,
        image_area * 0.00001
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 8 or h < 8:
            continue

        boxes.append(
            (x, y, w, h)
        )

    boxes = sorted(
        boxes,
        key=lambda box: (
            box[1],
            box[0]
        )
    )

    old_marked = old_image.copy()
    new_marked = new_image.copy()

    for index, (x, y, w, h) in enumerate(
        boxes,
        start=1
    ):

        padding = 10

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(
            old_image.shape[1] - 1,
            x + w + padding
        )

        y2 = min(
            old_image.shape[0] - 1,
            y + h + padding
        )

        cv2.rectangle(
            old_marked,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.rectangle(
            new_marked,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            old_marked,
            str(index),
            (x1, max(30, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            new_marked,
            str(index),
            (x1, max(30, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

    return (
        old_marked,
        new_marked,
        thresh,
        boxes
    )


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

    (
        old_marked,
        new_marked,
        diff_mask,
        boxes
    ) = find_differences(
        old_image,
        new_image
    )

    rows = [
        {
            "id": 0,
            "name": "偵測到的圖面差異區域",
            "old": 0,
            "new": len(boxes),
            "diff": len(boxes),
        }
    ]

    return jsonify(
        rows=rows,
        difference_count=len(boxes),
        old_image=encode_image(old_marked),
        new_image=encode_image(new_marked),
        diff_image=encode_image(diff_mask)
    )


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

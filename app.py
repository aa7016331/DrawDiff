import os
import csv
import io
import base64

from flask import Flask, render_template, request, jsonify, Response

import cv2
import numpy as np
import fitz


ALLOWED = {
    "png", "jpg", "jpeg",
    "bmp", "tif", "tiff",
    "pdf"
}

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024


def allowed(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED
    )


def read_upload(file):
    """
    讀取圖片或 PDF。
    PDF 目前先讀取第一頁。
    """

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

        image_bytes = pix.tobytes("png")

        doc.close()

        arr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        return cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR
        )

    arr = np.frombuffer(
        raw,
        np.uint8
    )

    return cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )


def image_to_base64(image):
    """
    OpenCV 圖片轉成可在網頁顯示的 Base64。
    """

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92
        ]
    )

    if not success:
        return None

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def resize_for_compare(old_image, new_image):
    """
    將新圖調整到與原圖相同尺寸。

    以原圖尺寸為基準。
    """

    old_h, old_w = old_image.shape[:2]

    new_resized = cv2.resize(
        new_image,
        (old_w, old_h),
        interpolation=cv2.INTER_AREA
    )

    return old_image, new_resized


def align_images(old_image, new_image):
    """
    嘗試用 ORB 特徵點做簡單對齊。

    如果對齊失敗，
    就直接使用 resize 後的圖片。
    """

    gray_old = cv2.cvtColor(
        old_image,
        cv2.COLOR_BGR2GRAY
    )

    gray_new = cv2.cvtColor(
        new_image,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        1500
    )

    kp1, des1 = orb.detectAndCompute(
        gray_old,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray_new,
        None
    )

    if (
        des1 is None
        or des2 is None
        or len(kp1) < 10
        or len(kp2) < 10
    ):
        return new_image

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(
        des1,
        des2
    )

    if len(matches) < 10:
        return new_image

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    good_matches = matches[
        :min(200, len(matches))
    ]

    if len(good_matches) < 10:
        return new_image

    src_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    matrix, mask = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC
    )

    if matrix is None:
        return new_image

    height, width = old_image.shape[:2]

    aligned = cv2.warpAffine(
        new_image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned


def find_differences(old_image, new_image):
    """
    找出兩張圖不同的位置。
    """

    gray_old = cv2.cvtColor(
        old_image,
        cv2.COLOR_BGR2GRAY
    )

    gray_new = cv2.cvtColor(
        new_image,
        cv2.COLOR_BGR2GRAY
    )

    gray_old = cv2.GaussianBlur(
        gray_old,
        (3, 3),
        0
    )

    gray_new = cv2.GaussianBlur(
        gray_new,
        (3, 3),
        0
    )

    diff = cv2.absdiff(
        gray_old,
        gray_new
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
        iterations=1
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    height, width = old_image.shape[:2]

    image_area = height * width

    min_area = max(
        80,
        image_area * 0.00002
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        boxes.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area)
            }
        )

    boxes = sorted(
        boxes,
        key=lambda box: (
            box["y"],
            box["x"]
        )
    )

    return boxes, thresh


def draw_differences(image, boxes):
    """
    在圖片上畫紅框與編號。
    """

    output = image.copy()

    for index, box in enumerate(
        boxes,
        start=1
    ):

        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        padding = 10

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(
            output.shape[1] - 1,
            x + w + padding
        )

        y2 = min(
            output.shape[0] - 1,
            y + h + padding
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        label = str(index)

        cv2.circle(
            output,
            (x1 + 18, y1 + 18),
            18,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            label,
            (x1 + 11, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


@app.get("/")
def index():

    return render_template(
        "index.html"
    )


@app.post("/analyze")
def analyze_api():

    old_file = request.files.get("old")
    new_file = request.files.get("new")

    if not old_file or not new_file:

        return jsonify(
            error="請同時上傳原始圖與修改後圖"
        ), 400

    if (
        not allowed(old_file.filename)
        or not allowed(new_file.filename)
    ):

        return jsonify(
            error="僅支援 PNG、JPG、JPEG、BMP、TIF、TIFF、PDF"
        ), 400

    old_image = read_upload(
        old_file
    )

    new_image = read_upload(
        new_file
    )

    if (
        old_image is None
        or new_image is None
    ):

        return jsonify(
            error="有圖片無法讀取"
        ), 400

    old_image, new_image = resize_for_compare(
        old_image,
        new_image
    )

    new_aligned = align_images(
        old_image,
        new_image
    )

    boxes, diff_mask = find_differences(
        old_image,
        new_aligned
    )

    old_marked = draw_differences(
        old_image,
        boxes
    )

    new_marked = draw_differences(
        new_aligned,
        boxes
    )

    rows = []

    for index, box in enumerate(
        boxes,
        start=1
    ):

        rows.append(
            {
                "id": index,
                "name": f"差異 {index}",
                "x": box["x"],
                "y": box["y"],
                "width": box["w"],
                "height": box["h"],
                "area": box["area"]
            }
        )

    return jsonify(
        {
            "count": len(boxes),
            "rows": rows,
            "old_image": image_to_base64(
                old_marked
            ),
            "new_image": image_to_base64(
                new_marked
            ),
            "diff_image": image_to_base64(
                diff_mask
            )
        }
    )


@app.post("/export")
def export_csv():

    data = request.get_json()

    rows = data.get(
        "rows",
        []
    )

    output = io.StringIO()

    writer = csv.writer(
        output
    )

    writer.writerow(
        [
            "編號",
            "差異項目",
            "X座標",
            "Y座標",
            "寬度",
            "高度",
            "差異面積"
        ]
    )

    for row in rows:

        writer.writerow(
            [
                row.get("id", ""),
                row.get("name", ""),
                row.get("x", ""),
                row.get("y", ""),
                row.get("width", ""),
                row.get("height", ""),
                row.get("area", "")
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
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )

import os
import csv
import io
import base64
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import fitz


BASE = Path(__file__).resolve().parent

ALLOWED = {
    "png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024


# ============================================================
# 基本工具
# ============================================================

def allowed(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED
    )


def read_upload(file):
    """
    讀取圖片或 PDF。
    PDF 目前先使用第一頁。
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

        png_bytes = pix.tobytes("png")

        doc.close()

        arr = np.frombuffer(
            png_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR
        )

        return image

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
    OpenCV 圖片轉成前端可顯示的 Base64。
    """

    ok, buffer = cv2.imencode(
        ".png",
        image
    )

    if not ok:
        return None

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# 裁掉白邊
# ============================================================

def crop_white_border(image):
    """
    移除圖面外圍大量白色空白。
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 非接近純白的區域
    mask = gray < 245

    coords = np.column_stack(
        np.where(mask)
    )

    if len(coords) < 100:
        return image

    y, x, h, w = cv2.boundingRect(
        coords.astype(np.int32)
    )

    # 加一些邊界，避免裁太緊
    padding = 15

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(
        image.shape[1],
        x + w + padding
    )

    y2 = min(
        image.shape[0],
        y + h + padding
    )

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# 預處理
# ============================================================

def preprocess_for_alignment(image):

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


# ============================================================
# 自動對齊
# ============================================================

def align_images(old_image, new_image):
    """
    使用 ORB + BFMatcher + RANSAC
    自動尋找兩張圖的共同特徵，
    將修改後圖對齊到原始圖。
    """

    old_gray = preprocess_for_alignment(
        old_image
    )

    new_gray = preprocess_for_alignment(
        new_image
    )

    orb = cv2.ORB_create(
        nfeatures=6000,
        scaleFactor=1.2,
        nlevels=8
    )

    kp1, des1 = orb.detectAndCompute(
        old_gray,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        new_gray,
        None
    )

    # 找不到足夠特徵時
    if (
        des1 is None
        or des2 is None
        or len(kp1) < 10
        or len(kp2) < 10
    ):

        resized = cv2.resize(
            new_image,
            (
                old_image.shape[1],
                old_image.shape[0]
            ),
            interpolation=cv2.INTER_CUBIC
        )

        return resized, np.eye(3)


    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    raw_matches = matcher.knnMatch(
        des2,
        des1,
        k=2
    )

    good = []

    for pair in raw_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        # Lowe ratio test
        if m.distance < 0.75 * n.distance:
            good.append(m)


    # 如果匹配點太少
    if len(good) < 8:

        resized = cv2.resize(
            new_image,
            (
                old_image.shape[1],
                old_image.shape[0]
            ),
            interpolation=cv2.INTER_CUBIC
        )

        return resized, np.eye(3)


    src_points = np.float32(
        [
            kp2[m.queryIdx].pt
            for m in good
        ]
    ).reshape(
        -1,
        1,
        2
    )

    dst_points = np.float32(
        [
            kp1[m.trainIdx].pt
            for m in good
        ]
    ).reshape(
        -1,
        1,
        2
    )


    # 使用 Affine，比 Homography 更不容易因局部文字
    # 產生過度變形
    matrix, inliers = cv2.estimateAffinePartial2D(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=5000,
        confidence=0.995
    )


    if matrix is None:

        resized = cv2.resize(
            new_image,
            (
                old_image.shape[1],
                old_image.shape[0]
            ),
            interpolation=cv2.INTER_CUBIC
        )

        return resized, np.eye(3)


    aligned_new = cv2.warpAffine(
        new_image,
        matrix,
        (
            old_image.shape[1],
            old_image.shape[0]
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


    # 回傳 3x3 方便之後擴充
    matrix_3x3 = np.eye(3)
    matrix_3x3[:2, :] = matrix

    return aligned_new, matrix_3x3


# ============================================================
# 差異分析
# ============================================================

def find_differences(old_image, aligned_new):

    old_gray = cv2.cvtColor(
        old_image,
        cv2.COLOR_BGR2GRAY
    )

    new_gray = cv2.cvtColor(
        aligned_new,
        cv2.COLOR_BGR2GRAY
    )


    # 輕微模糊，降低 JPEG 雜訊
    old_blur = cv2.GaussianBlur(
        old_gray,
        (3, 3),
        0
    )

    new_blur = cv2.GaussianBlur(
        new_gray,
        (3, 3),
        0
    )


    # 絕對差異
    diff = cv2.absdiff(
        old_blur,
        new_blur
    )


    # 只保留較明顯差異
    _, mask = cv2.threshold(
        diff,
        45,
        255,
        cv2.THRESH_BINARY
    )


    # 消除非常細小的雜訊
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1
    )


    # 將同一個修改區域連接起來
    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (11, 11)
    )

    mask = cv2.dilate(
        mask,
        connect_kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        connect_kernel,
        iterations=2
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    differences = []

    image_area = (
        old_image.shape[0]
        * old_image.shape[1]
    )


    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = w * h

        # 太小通常是雜訊
        if area < 150:
            continue

        # 太大的通常表示對齊失敗或整張圖偏差
        if area > image_area * 0.60:
            continue


        differences.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area)
            }
        )


    # 由上到下、左到右排序
    differences.sort(
        key=lambda d: (
            d["y"],
            d["x"]
        )
    )


    # 合併彼此非常接近的框
    merged = []

    for box in differences:

        bx1 = box["x"]
        by1 = box["y"]
        bx2 = bx1 + box["w"]
        by2 = by1 + box["h"]

        merged_into_existing = False

        for existing in merged:

            ex1 = existing["x"] - 15
            ey1 = existing["y"] - 15

            ex2 = (
                existing["x"]
                + existing["w"]
                + 15
            )

            ey2 = (
                existing["y"]
                + existing["h"]
                + 15
            )

            overlap = not (
                bx2 < ex1
                or bx1 > ex2
                or by2 < ey1
                or by1 > ey2
            )

            if overlap:

                nx1 = min(
                    existing["x"],
                    bx1
                )

                ny1 = min(
                    existing["y"],
                    by1
                )

                nx2 = max(
                    existing["x"]
                    + existing["w"],
                    bx2
                )

                ny2 = max(
                    existing["y"]
                    + existing["h"],
                    by2
                )

                existing["x"] = nx1
                existing["y"] = ny1

                existing["w"] = nx2 - nx1
                existing["h"] = ny2 - ny1

                existing["area"] = (
                    existing["w"]
                    * existing["h"]
                )

                merged_into_existing = True

                break


        if not merged_into_existing:
            merged.append(box)


    return mask, merged


# ============================================================
# 在圖片上畫差異框
# ============================================================

def draw_difference_boxes(image, differences):

    result = image.copy()

    for index, diff in enumerate(
        differences,
        start=1
    ):

        x = diff["x"]
        y = diff["y"]
        w = diff["w"]
        h = diff["h"]

        # 紅框
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )


        # 編號背景圓
        center_x = x + 12
        center_y = max(15, y + 12)

        cv2.circle(
            result,
            (
                center_x,
                center_y
            ),
            13,
            (0, 0, 255),
            -1
        )


        # 白色編號
        cv2.putText(
            result,
            str(index),
            (
                center_x - 5,
                center_y + 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    return result


# ============================================================
# 首頁
# ============================================================

@app.get("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# 分析 API
# ============================================================

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
            error="圖片無法讀取"
        ), 400


    # --------------------------------------------------------
    # 1. 裁白邊
    # --------------------------------------------------------

    old_image = crop_white_border(
        old_image
    )

    new_image = crop_white_border(
        new_image
    )


    # --------------------------------------------------------
    # 2. 自動對齊
    # --------------------------------------------------------

    aligned_new, matrix = align_images(
        old_image,
        new_image
    )


    # --------------------------------------------------------
    # 3. 找真正差異
    # --------------------------------------------------------

    mask, differences = find_differences(
        old_image,
        aligned_new
    )


    # --------------------------------------------------------
    # 4. 左右同步畫框
    # --------------------------------------------------------

    old_marked = draw_difference_boxes(
        old_image,
        differences
    )

    new_marked = draw_difference_boxes(
        aligned_new,
        differences
    )


    rows = []

    for index, diff in enumerate(
        differences,
        start=1
    ):

        rows.append(
            {
                "id": index,
                "x": diff["x"],
                "y": diff["y"],
                "width": diff["w"],
                "height": diff["h"],
                "area": diff["area"]
            }
        )


    return jsonify(
        difference_count=len(differences),

        old_image=image_to_base64(
            old_marked
        ),

        new_image=image_to_base64(
            new_marked
        ),

        mask=image_to_base64(
            mask
        ),

        rows=rows
    )


# ============================================================
# 匯出 CSV
# ============================================================

@app.post("/export")
def export_csv():

    data = request.get_json() or {}

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
            "差異編號",
            "X",
            "Y",
            "寬度",
            "高度",
            "差異面積"
        ]
    )


    for row in rows:

        writer.writerow(
            [
                row.get("id", ""),
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


# ============================================================
# 啟動
# ============================================================

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

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
    PDF 目前使用第一頁。
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
    OpenCV 圖片轉成前端可顯示的 Base64。
    """

    if image is None:
        return None

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

    mask = gray < 245

    coords = np.column_stack(
        np.where(mask)
    )

    if len(coords) < 100:
        return image

    y, x, h, w = cv2.boundingRect(
        coords.astype(np.int32)
    )

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
    將修改後圖對齊到原始圖。
    """

    old_gray = preprocess_for_alignment(
        old_image
    )

    new_gray = preprocess_for_alignment(
        new_image
    )

    orb = cv2.ORB_create(
        nfeatures=10000,
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

        return resized


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

        if m.distance < 0.72 * n.distance:
            good.append(m)


    if len(good) < 10:

        resized = cv2.resize(
            new_image,
            (
                old_image.shape[1],
                old_image.shape[0]
            ),
            interpolation=cv2.INTER_CUBIC
        )

        return resized


    good = sorted(
        good,
        key=lambda m: m.distance
    )[:2500]


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


    matrix, inliers = cv2.estimateAffinePartial2D(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=10000,
        confidence=0.999
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

        return resized


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


    return aligned_new


# ============================================================
# 細部對齊
# ============================================================

def refine_translation(old_image, new_image):
    """
    ORB 對齊後，再用 phase correlation
    做細微平移修正。

    這一步主要是避免圖面只差 1~3 像素，
    卻造成整張圖被判定為差異。
    """

    old_gray = cv2.cvtColor(
        old_image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    new_gray = cv2.cvtColor(
        new_image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    old_gray = cv2.GaussianBlur(
        old_gray,
        (5, 5),
        0
    )

    new_gray = cv2.GaussianBlur(
        new_gray,
        (5, 5),
        0
    )

    try:

        shift, response = cv2.phaseCorrelate(
            old_gray,
            new_gray
        )

    except cv2.error:

        return new_image


    dx, dy = shift


    # 只有可信且很小的微調才採用
    if (
        response < 0.05
        or abs(dx) > 20
        or abs(dy) > 20
    ):
        return new_image


    matrix = np.float32(
        [
            [1, 0, dx],
            [0, 1, dy]
        ]
    )


    return cv2.warpAffine(
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


# ============================================================
# 找局部真正差異
# ============================================================

def find_differences(old_image, aligned_new):
    """
    不再直接把整張圖的灰階 absdiff 拿來圈。

    新邏輯：

    1. 先把兩張圖轉成二值圖。
    2. 做小範圍容錯，避免線條因 1~2 像素偏移被判定。
    3. 找出「一張有墨跡、另一張附近沒有」的局部區域。
    4. 排除細長線、文字碎片與過大的區域。
    5. 合併同一設備附近的差異。
    """

    old_gray = cv2.cvtColor(
        old_image,
        cv2.COLOR_BGR2GRAY
    )

    new_gray = cv2.cvtColor(
        aligned_new,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # 自適應二值化
    # --------------------------------------------------------

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


    _, old_ink = cv2.threshold(
        old_blur,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    _, new_ink = cv2.threshold(
        new_blur,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )


    # --------------------------------------------------------
    # 容忍微小對齊誤差
    #
    # 將線條稍微膨脹後再比較：
    # 兩張圖相同位置附近都有線，就不算差異。
    # --------------------------------------------------------

    tolerance_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    old_near = cv2.dilate(
        old_ink,
        tolerance_kernel,
        iterations=1
    )

    new_near = cv2.dilate(
        new_ink,
        tolerance_kernel,
        iterations=1
    )


    # 原始圖有、新圖附近沒有
    removed = cv2.bitwise_and(
        old_ink,
        cv2.bitwise_not(new_near)
    )


    # 新圖有、原始圖附近沒有
    added = cv2.bitwise_and(
        new_ink,
        cv2.bitwise_not(old_near)
    )


    mask = cv2.bitwise_or(
        removed,
        added
    )


    # --------------------------------------------------------
    # 去除非常細小的雜訊
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 把同一個設備的小筆畫連起來
    #
    # 注意：
    # 不再使用很大的 11x11 + 多次 CLOSE，
    # 避免整張圖的線條全部連成大區塊。
    # --------------------------------------------------------

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
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
        iterations=1
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []

    image_h, image_w = old_image.shape[:2]
    image_area = image_h * image_w


    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = w * h


        # ----------------------------------------------------
        # 基本大小過濾
        # ----------------------------------------------------

        if area < 120:
            continue

        if area > image_area * 0.08:
            continue


        # 太細長的通常是尺寸線、牆線、文字線
        aspect = max(
            w / max(h, 1),
            h / max(w, 1)
        )

        if aspect > 12:
            continue


        # ----------------------------------------------------
        # 計算真正差異像素比例
        #
        # 外框很大但裡面只有幾條線，
        # 很可能不是一個真正設備修改。
        # ----------------------------------------------------

        roi = mask[
            y:y + h,
            x:x + w
        ]

        changed_pixels = cv2.countNonZero(
            roi
        )

        fill_ratio = changed_pixels / max(
            area,
            1
        )


        if fill_ratio < 0.02:
            continue


        # ----------------------------------------------------
        # 過小的文字碎片排除
        # ----------------------------------------------------

        if w < 8 or h < 8:
            continue


        candidates.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
        )


    # ========================================================
    # 合併靠近的候選區域
    # ========================================================

    merged = []


    for box in candidates:

        bx1 = box["x"]
        by1 = box["y"]
        bx2 = bx1 + box["w"]
        by2 = by1 + box["h"]

        merged_index = None


        for i, existing in enumerate(merged):

            ex1 = existing["x"]
            ey1 = existing["y"]
            ex2 = ex1 + existing["w"]
            ey2 = ey1 + existing["h"]


            # 允許設備附近的小差異合併
            padding = 18


            overlap = not (
                bx2 < ex1 - padding
                or bx1 > ex2 + padding
                or by2 < ey1 - padding
                or by1 > ey2 + padding
            )


            if overlap:

                merged_index = i
                break


        if merged_index is None:

            merged.append(
                box
            )

        else:

            existing = merged[
                merged_index
            ]

            nx1 = min(
                existing["x"],
                bx1
            )

            ny1 = min(
                existing["y"],
                by1
            )

            nx2 = max(
                existing["x"] + existing["w"],
                bx2
            )

            ny2 = max(
                existing["y"] + existing["h"],
                by2
            )


            existing["x"] = int(nx1)
            existing["y"] = int(ny1)
            existing["w"] = int(nx2 - nx1)
            existing["h"] = int(ny2 - ny1)


    # ========================================================
    # 第二次過濾
    #
    # 合併後太大的區域通常仍是圖面整體誤差
    # ========================================================

    final_differences = []


    for box in merged:

        area = (
            box["w"]
            * box["h"]
        )


        if area > image_area * 0.04:
            continue


        # 避免框變成非常長的牆線
        aspect = max(
            box["w"] / max(box["h"], 1),
            box["h"] / max(box["w"], 1)
        )

        if aspect > 10:
            continue


        # 加一些外框邊界，讓設備比較容易看清楚
        padding = 8

        x1 = max(
            0,
            box["x"] - padding
        )

        y1 = max(
            0,
            box["y"] - padding
        )

        x2 = min(
            image_w,
            box["x"] + box["w"] + padding
        )

        y2 = min(
            image_h,
            box["y"] + box["h"] + padding
        )


        final_differences.append(
            {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1)
            }
        )


    # 由上到下、左到右排序
    final_differences.sort(
        key=lambda d: (
            d["y"],
            d["x"]
        )
    )


    return mask, final_differences


# ============================================================
# 在圖片上畫同步差異圈
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


        # ----------------------------------------------------
        # 使用圓圈，而不是大紅色矩形
        # ----------------------------------------------------

        center_x = x + w // 2
        center_y = y + h // 2

        radius = max(
            14,
            int(
                max(w, h) * 0.65
            )
        )


        # 紅色圈圈
        cv2.circle(
            result,
            (
                center_x,
                center_y
            ),
            radius,
            (0, 0, 255),
            3
        )


        # ----------------------------------------------------
        # 編號紅色圓點
        # ----------------------------------------------------

        label_x = max(
            14,
            center_x - radius
        )

        label_y = max(
            14,
            center_y - radius
        )


        cv2.circle(
            result,
            (
                label_x,
                label_y
            ),
            14,
            (0, 0, 255),
            -1
        )


        text = str(index)

        text_size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        text_x = int(
            label_x
            - text_size[0] / 2
        )

        text_y = int(
            label_y
            + text_size[1] / 2
        )


        cv2.putText(
            result,
            text,
            (
                text_x,
                text_y
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
    # 1. 裁掉外圍白邊
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

    aligned_new = align_images(
        old_image,
        new_image
    )


    # --------------------------------------------------------
    # 3. 微調平移
    # --------------------------------------------------------

    aligned_new = refine_translation(
        old_image,
        aligned_new
    )


    # --------------------------------------------------------
    # 4. 找局部真正差異
    # --------------------------------------------------------

    mask, differences = find_differences(
        old_image,
        aligned_new
    )


    # --------------------------------------------------------
    # 5. 左右同步畫圈
    # --------------------------------------------------------

    old_marked = draw_difference_boxes(
        old_image,
        differences
    )

    new_marked = draw_difference_boxes(
        aligned_new,
        differences
    )


    # --------------------------------------------------------
    # 6. 建立資料列
    # --------------------------------------------------------

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
                "height": diff["h"]
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

        # 暫時保留欄位，避免舊版 index.html 報錯
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
            "高度"
        ]
    )


    for row in rows:

        writer.writerow(
            [
                row.get("id", ""),
                row.get("x", ""),
                row.get("y", ""),
                row.get("width", ""),
                row.get("height", "")
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

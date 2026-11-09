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

# ============================================================
# 品項圖庫
# ============================================================

SYMBOL_DIR = BASE / "symbols"

SYMBOL_NAMES = {
    0: "住戶開關箱與弱電箱整合",
    1: "馬桶專用插座 H:30cm",
    2: "抽油煙機插座 H:天花板內",
    3: "雙聯插座 H:30cm，電視牆 H:80cm",
    4: "雙聯插座 H:110cm",
    5: "洗碗機專用插座 H:45cm",
    6: "微波烤箱專用插座 220V(H:55cm)",
    7: "洗衣機插座(防雨型) H:120cm",
    8: "廚房檯櫃燈用電源出線口 H:170cm",
    9: "暖風乾燥機電源出口 220V(H:天花)",
    10: "雙聯紅色緊急插座(弱電、箱體內)",
    11: "雙聯紅色緊急插座(冰箱 H:30cm、電視牆 H:80cm)",
    12: "冷氣電源出口 220V(H:天花板)"
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
# 載入品項圖案
# ============================================================

def load_symbol_templates():

    templates = {}

    for symbol_id in SYMBOL_NAMES:

        path = SYMBOL_DIR / f"{symbol_id}.png"

        if not path.exists():
            continue

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue


        # ----------------------------------------------------
        # 去除純白邊
        # ----------------------------------------------------

        _, binary = cv2.threshold(
            image,
            245,
            255,
            cv2.THRESH_BINARY_INV
        )

        coords = cv2.findNonZero(
            binary
        )

        if coords is not None:

            x, y, w, h = cv2.boundingRect(
                coords
            )

            padding = 3

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

            image = image[
                y1:y2,
                x1:x2
            ]


        templates[symbol_id] = image


    return templates


# ============================================================
# 品項辨識
# ============================================================

def classify_symbol(roi, templates):

    """
    將差異區域與 0~12.png 圖庫進行比對。

    回傳：
    symbol_id
    symbol_name
    confidence
    """

    if roi is None or roi.size == 0:

        return None, "未辨識", 0.0


    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # ROI 二值化
    # --------------------------------------------------------

    _, roi_binary = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )


    # 稍微去除雜訊
    roi_binary = cv2.morphologyEx(
        roi_binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2, 2)
        )
    )


    best_id = None
    best_score = -1.0


    for symbol_id, template in templates.items():

        if template is None:
            continue


        # ----------------------------------------------------
        # Template 二值化
        # ----------------------------------------------------

        _, template_binary = cv2.threshold(
            template,
            200,
            255,
            cv2.THRESH_BINARY_INV
        )


        template_h, template_w = (
            template_binary.shape[:2]
        )


        roi_h, roi_w = (
            roi_binary.shape[:2]
        )


        if (
            template_h < 5
            or template_w < 5
        ):
            continue


        # ----------------------------------------------------
        # 多尺寸比對
        #
        # 因為圖面中的設備符號
        # 可能比原始圖庫大或小
        # ----------------------------------------------------

        for scale in np.linspace(
            0.5,
            2.5,
            13
        ):

            resized_w = int(
                template_w * scale
            )

            resized_h = int(
                template_h * scale
            )


            if (
                resized_w < 5
                or resized_h < 5
            ):
                continue


            if (
                resized_w > roi_w
                or resized_h > roi_h
            ):
                continue


            resized_template = cv2.resize(
                template_binary,
                (
                    resized_w,
                    resized_h
                ),
                interpolation=cv2.INTER_AREA
            )


            try:

                result = cv2.matchTemplate(
                    roi_binary,
                    resized_template,
                    cv2.TM_CCOEFF_NORMED
                )

                _, max_val, _, _ = cv2.minMaxLoc(
                    result
                )

            except cv2.error:

                continue


            if max_val > best_score:

                best_score = float(
                    max_val
                )

                best_id = symbol_id


    # --------------------------------------------------------
    # 信心值門檻
    #
    # 目前先設 0.42
    # 之後可以依實際圖面再調整
    # ----------------------------------------------------

    if (
        best_id is None
        or best_score < 0.42
    ):

        return (
            None,
            "未辨識",
            round(best_score, 3)
        )


    return (
        best_id,
        SYMBOL_NAMES.get(
            best_id,
            "未辨識"
        ),
        round(best_score, 3)
    )


# ============================================================
# 裁掉白邊
# ============================================================

def crop_white_border(image):

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

        return cv2.resize(
            new_image,
            (
                old_image.shape[1],
                old_image.shape[0]
            ),
            interpolation=cv2.INTER_CUBIC
        )


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

        return cv2.resize(
            new_image,
            (
                old_image.shape[1],
                old_image.shape[0]
            ),
            interpolation=cv2.INTER_CUBIC
        )


    good = sorted(
        good,
        key=lambda m: m.distance
    )[:2500]


    src_points = np.float32(
        [
            kp2[m.queryIdx].pt
            for m in good
        ]
    ).reshape(-1, 1, 2)


    dst_points = np.float32(
        [
            kp1[m.trainIdx].pt
            for m in good
        ]
    ).reshape(-1, 1, 2)


    matrix, inliers = cv2.estimateAffinePartial2D(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=10000,
        confidence=0.999
    )


    if matrix is None:

        return cv2.resize(
            new_image,
            (
                old_image.shape[1],
                old_image.shape[0]
            ),
            interpolation=cv2.INTER_CUBIC
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
# 細部平移修正
# ============================================================

def refine_translation(old_image, new_image):

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
# 找真正差異
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


    removed = cv2.bitwise_and(
        old_ink,
        cv2.bitwise_not(new_near)
    )

    added = cv2.bitwise_and(
        new_ink,
        cv2.bitwise_not(old_near)
    )


    mask = cv2.bitwise_or(
        removed,
        added
    )


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


        if area < 120:
            continue

        if area > image_area * 0.08:
            continue


        aspect = max(
            w / max(h, 1),
            h / max(w, 1)
        )

        if aspect > 12:
            continue


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


    # --------------------------------------------------------
    # 合併附近差異
    # --------------------------------------------------------

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


    final_differences = []


    for box in merged:

        area = (
            box["w"]
            * box["h"]
        )


        if area > image_area * 0.04:
            continue


        aspect = max(
            box["w"] / max(box["h"], 1),
            box["h"] / max(box["w"], 1)
        )

        if aspect > 10:
            continue


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


    final_differences.sort(
        key=lambda d: (
            d["y"],
            d["x"]
        )
    )


    return mask, final_differences


# ============================================================
# 畫同步差異圈
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


        center_x = x + w // 2
        center_y = y + h // 2


        radius = max(
            14,
            int(max(w, h) * 0.65)
        )


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
    # 1. 裁掉白邊
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
    # 3. 微調
    # --------------------------------------------------------

    aligned_new = refine_translation(
        old_image,
        aligned_new
    )


    # --------------------------------------------------------
    # 4. 找差異
    # --------------------------------------------------------

    mask, differences = find_differences(
        old_image,
        aligned_new
    )


    # --------------------------------------------------------
    # 5. 載入品項圖庫
    # --------------------------------------------------------

    symbol_templates = load_symbol_templates()


    # --------------------------------------------------------
    # 6. 辨識每個差異是什麼品項
    # --------------------------------------------------------

    rows = []


    image_h, image_w = old_image.shape[:2]


    for index, diff in enumerate(
        differences,
        start=1
    ):

        x = diff["x"]
        y = diff["y"]
        w = diff["w"]
        h = diff["h"]


        # ----------------------------------------------------
        # 擴大辨識範圍
        #
        # 差異框本身可能只抓到部分線條，
        # 所以額外往外取一圈
        # ----------------------------------------------------

        recognize_padding = 25

        x1 = max(
            0,
            x - recognize_padding
        )

        y1 = max(
            0,
            y - recognize_padding
        )

        x2 = min(
            image_w,
            x + w + recognize_padding
        )

        y2 = min(
            image_h,
            y + h + recognize_padding
        )


        old_roi = old_image[
            y1:y2,
            x1:x2
        ]

        new_roi = aligned_new[
            y1:y2,
            x1:x2
        ]


        # ----------------------------------------------------
        # 原圖辨識
        # ----------------------------------------------------

        old_symbol_id, old_symbol_name, old_score = (
            classify_symbol(
                old_roi,
                symbol_templates
            )
        )


        # ----------------------------------------------------
        # 新圖辨識
        # ----------------------------------------------------

        new_symbol_id, new_symbol_name, new_score = (
            classify_symbol(
                new_roi,
                symbol_templates
            )
        )


        # ----------------------------------------------------
        # 取信心值較高的辨識結果
        # ----------------------------------------------------

        if new_score >= old_score:

            symbol_id = new_symbol_id
            symbol_name = new_symbol_name
            confidence = new_score

        else:

            symbol_id = old_symbol_id
            symbol_name = old_symbol_name
            confidence = old_score


        rows.append(
            {
                "id": index,

                "x": x,
                "y": y,

                "width": w,
                "height": h,

                "symbol_id": (
                    symbol_id
                    if symbol_id is not None
                    else ""
                ),

                "symbol_name": symbol_name,

                "confidence": confidence
            }
        )


    # --------------------------------------------------------
    # 7. 畫圈
    # --------------------------------------------------------

    old_marked = draw_difference_boxes(
        old_image,
        differences
    )

    new_marked = draw_difference_boxes(
        aligned_new,
        differences
    )


    return jsonify(

        difference_count=len(
            differences
        ),

        old_image=image_to_base64(
            old_marked
        ),

        new_image=image_to_base64(
            new_marked
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
            "品項編號",
            "品項名稱",
            "辨識信心值"
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
                row.get("symbol_id", ""),
                row.get("symbol_name", ""),
                row.get("confidence", "")
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

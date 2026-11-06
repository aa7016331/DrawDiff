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
SYMBOL_DIR = BASE / "static" / "symbols"

ALLOWED = {
    "png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024


# ============================================================
# 設備圖例
# ============================================================
#
# 這裡的順序對應：
#
# static/symbols/0.png
# static/symbols/1.png
# ...
# static/symbols/12.png
#
# 如果你的切圖順序和實際圖例順序不同，
# 之後只需要修改這裡的名稱。
# ============================================================

SYMBOL_NAMES = {
    "0": "住戶開關箱與弱電箱整合",
    "1": "馬桶專用插座 H:30cm",
    "2": "抽油煙機插座 H:天花板內",
    "3": "雙聯插座 H:30cm",
    "4": "雙聯插座 H:110cm",
    "5": "洗碗機專用插座 H:45cm",
    "6": "微波烤箱專用插座 220V(H:55cm)",
    "7": "洗衣機插座(防雨型) H:120cm",
    "8": "廚房橫櫃燈用電源出線口 H:170cm",
    "9": "暖風乾燥機電源出口 220V(H:天花)",
    "10": "雙聯紅色緊急插座(弱電、箱體內)",
    "11": "雙聯紅色緊急插座(冰箱 H:30cm、電視牆 H:80cm)",
    "12": "冷氣電源出口 220V(H:天花板)",
}


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
# 自動對齊前處理
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
    將修改後圖自動對齊到原圖。
    """

    old_gray = preprocess_for_alignment(
        old_image
    )

    new_gray = preprocess_for_alignment(
        new_image
    )

    orb = cv2.ORB_create(
        nfeatures=8000,
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

        if m.distance < 0.75 * n.distance:
            good.append(m)


    if len(good) < 8:

        resized = cv2.resize(
            new_image,
            (
                old_image.shape[1],
                old_image.shape[0]
            ),
            interpolation=cv2.INTER_CUBIC
        )

        return resized


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
# 載入設備符號
# ============================================================

def load_symbol_templates():
    """
    讀取 static/symbols/0.png ~ 12.png
    """

    templates = {}

    if not SYMBOL_DIR.exists():
        return templates

    for symbol_id, symbol_name in SYMBOL_NAMES.items():

        file_path = SYMBOL_DIR / f"{symbol_id}.png"

        if not file_path.exists():
            continue

        template = cv2.imread(
            str(file_path),
            cv2.IMREAD_GRAYSCALE
        )

        if template is None:
            continue

        # 去除圖片外圍空白
        template = crop_template(
            template
        )

        if (
            template is None
            or template.shape[0] < 5
            or template.shape[1] < 5
        ):
            continue

        templates[symbol_id] = {
            "name": symbol_name,
            "image": template
        }

    return templates


def crop_template(template):
    """
    裁掉設備範本外圍白色區域。
    """

    if len(template.shape) == 3:

        gray = cv2.cvtColor(
            template,
            cv2.COLOR_BGR2GRAY
        )

    else:
        gray = template.copy()


    mask = gray < 245

    coords = np.column_stack(
        np.where(mask)
    )

    if len(coords) < 5:
        return gray

    y, x, h, w = cv2.boundingRect(
        coords.astype(np.int32)
    )

    padding = 2

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(
        gray.shape[1],
        x + w + padding
    )

    y2 = min(
        gray.shape[0],
        y + h + padding
    )

    return gray[
        y1:y2,
        x1:x2
    ]


# ============================================================
# 非極大值抑制
# ============================================================

def non_max_suppression(detections, overlap_threshold=0.35):
    """
    移除同一個設備被重複辨識的情況。
    """

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda d: d["score"],
        reverse=True
    )

    kept = []

    while detections:

        best = detections.pop(0)

        kept.append(best)

        remaining = []

        bx1 = best["x"]
        by1 = best["y"]

        bx2 = bx1 + best["w"]
        by2 = by1 + best["h"]


        for candidate in detections:

            cx1 = candidate["x"]
            cy1 = candidate["y"]

            cx2 = cx1 + candidate["w"]
            cy2 = cy1 + candidate["h"]


            ix1 = max(bx1, cx1)
            iy1 = max(by1, cy1)

            ix2 = min(bx2, cx2)
            iy2 = min(by2, cy2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)

            intersection = iw * ih

            best_area = best["w"] * best["h"]
            candidate_area = (
                candidate["w"]
                * candidate["h"]
            )

            union = (
                best_area
                + candidate_area
                - intersection
            )

            iou = (
                intersection / union
                if union > 0
                else 0
            )

            if iou < overlap_threshold:
                remaining.append(candidate)


        detections = remaining


    return kept


# ============================================================
# 設備符號辨識
# ============================================================

def detect_symbol(
    image,
    template,
    threshold=0.58
):
    """
    使用多倍率 Template Matching
    尋找設備符號。

    因為上傳圖可能縮放，
    所以會測試多個尺寸。
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    template = cv2.GaussianBlur(
        template,
        (3, 3),
        0
    )


    detections = []

    # 多倍率搜尋
    scales = [
        0.70,
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
        1.35,
        1.50,
        1.70,
        2.00
    ]


    for scale in scales:

        resized_template = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        th, tw = resized_template.shape[:2]

        if (
            th >= gray.shape[0]
            or tw >= gray.shape[1]
        ):
            continue


        result = cv2.matchTemplate(
            gray,
            resized_template,
            cv2.TM_CCOEFF_NORMED
        )


        locations = np.where(
            result >= threshold
        )


        for y, x in zip(
            locations[0],
            locations[1]
        ):

            score = float(
                result[y, x]
            )

            detections.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(tw),
                    "h": int(th),
                    "score": score
                }
            )


    return non_max_suppression(
        detections,
        overlap_threshold=0.30
    )


# ============================================================
# 所有設備辨識
# ============================================================

def detect_all_symbols(
    image,
    templates
):
    """
    回傳：
    {
        "0": [...],
        "1": [...],
        ...
    }
    """

    all_results = {}

    for symbol_id, info in templates.items():

        detections = detect_symbol(
            image,
            info["image"]
        )

        all_results[symbol_id] = detections

    return all_results


# ============================================================
# 比較同一類設備位置
# ============================================================

def compare_symbol_positions(
    old_detections,
    new_detections,
    distance_threshold=45
):
    """
    已經自動對齊後：

    位置接近 = 同一個設備
    原圖有、新圖沒有 = 減少
    新圖有、原圖沒有 = 增加
    """

    unmatched_old = old_detections.copy()
    unmatched_new = new_detections.copy()

    matched_old = set()
    matched_new = set()


    # 依距離找最佳配對
    candidates = []

    for old_index, old_item in enumerate(
        unmatched_old
    ):

        old_cx = (
            old_item["x"]
            + old_item["w"] / 2
        )

        old_cy = (
            old_item["y"]
            + old_item["h"] / 2
        )


        for new_index, new_item in enumerate(
            unmatched_new
        ):

            new_cx = (
                new_item["x"]
                + new_item["w"] / 2
            )

            new_cy = (
                new_item["y"]
                + new_item["h"] / 2
            )


            distance = np.sqrt(
                (old_cx - new_cx) ** 2
                + (old_cy - new_cy) ** 2
            )


            candidates.append(
                (
                    distance,
                    old_index,
                    new_index
                )
            )


    candidates.sort(
        key=lambda item: item[0]
    )


    for distance, old_index, new_index in candidates:

        if distance > distance_threshold:
            break

        if old_index in matched_old:
            continue

        if new_index in matched_new:
            continue

        matched_old.add(
            old_index
        )

        matched_new.add(
            new_index
        )


    removed = []

    for index, item in enumerate(
        unmatched_old
    ):

        if index not in matched_old:
            removed.append(item)


    added = []

    for index, item in enumerate(
        unmatched_new
    ):

        if index not in matched_new:
            added.append(item)


    return added, removed


# ============================================================
# 畫設備差異圈
# ============================================================

def draw_symbol_marks(
    image,
    marks
):
    """
    marks 格式：

    [
        {
            "id": 1,
            "symbol_id": "3",
            "type": "add",
            "x": ...,
            "y": ...,
            "w": ...,
            "h": ...
        }
    ]
    """

    result = image.copy()

    for mark in marks:

        x = mark["x"]
        y = mark["y"]

        w = mark["w"]
        h = mark["h"]

        mark_id = mark["id"]

        center_x = int(
            x + w / 2
        )

        center_y = int(
            y + h / 2
        )

        radius = int(
            max(w, h) / 2
            + 12
        )


        # 紅色圓圈
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


        # 編號圓點位置
        label_x = center_x - radius
        label_y = center_y - radius

        label_x = max(
            15,
            label_x
        )

        label_y = max(
            15,
            label_y
        )


        # 編號紅色圓
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


        # 白色編號
        text = str(mark_id)

        cv2.putText(
            result,
            text,
            (
                label_x - 5,
                label_y + 6
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


    # --------------------------------------------------------
    # 讀取圖片
    # --------------------------------------------------------

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
    # 載入設備符號
    # --------------------------------------------------------

    templates = load_symbol_templates()

    if not templates:

        return jsonify(
            error=(
                "找不到設備符號範本。"
                "請確認 static/symbols/"
                "內已有 0.png ~ 12.png"
            )
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
    # 3. 辨識原圖設備
    # --------------------------------------------------------

    old_symbols = detect_all_symbols(
        old_image,
        templates
    )


    # --------------------------------------------------------
    # 4. 辨識修改後設備
    # --------------------------------------------------------

    new_symbols = detect_all_symbols(
        aligned_new,
        templates
    )


    # --------------------------------------------------------
    # 5. 比較設備
    # --------------------------------------------------------

    old_marks = []
    new_marks = []

    rows = []

    mark_number = 1


    total_added = 0
    total_removed = 0


    for symbol_id, info in templates.items():

        old_detections = old_symbols.get(
            symbol_id,
            []
        )

        new_detections = new_symbols.get(
            symbol_id,
            []
        )


        added, removed = compare_symbol_positions(
            old_detections,
            new_detections
        )


        # --------------------------------------------
        # 原圖減少的設備
        # --------------------------------------------

        for item in removed:

            old_marks.append(
                {
                    "id": mark_number,
                    "symbol_id": symbol_id,
                    "type": "remove",
                    "x": item["x"],
                    "y": item["y"],
                    "w": item["w"],
                    "h": item["h"]
                }
            )

            mark_number += 1


        # --------------------------------------------
        # 修改後增加的設備
        # --------------------------------------------

        for item in added:

            new_marks.append(
                {
                    "id": mark_number,
                    "symbol_id": symbol_id,
                    "type": "add",
                    "x": item["x"],
                    "y": item["y"],
                    "w": item["w"],
                    "h": item["h"]
                }
            )

            mark_number += 1


        # 只有數量不同才顯示
        if (
            len(old_detections)
            != len(new_detections)
        ):

            change = (
                len(new_detections)
                - len(old_detections)
            )

            rows.append(
                {
                    "symbol_id": symbol_id,
                    "equipment": info["name"],
                    "old_count": len(old_detections),
                    "new_count": len(new_detections),
                    "change": change
                }
            )


        total_added += len(added)
        total_removed += len(removed)


    # --------------------------------------------------------
    # 6. 畫紅圈
    # --------------------------------------------------------

    old_marked = draw_symbol_marks(
        old_image,
        old_marks
    )

    new_marked = draw_symbol_marks(
        aligned_new,
        new_marks
    )


    # --------------------------------------------------------
    # 回傳結果
    # --------------------------------------------------------

    return jsonify(

        added_count=total_added,

        removed_count=total_removed,

        difference_count=(
            total_added
            + total_removed
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


    # 不再輸出 X / Y / 面積
    writer.writerow(
        [
            "設備圖例",
            "設備名稱",
            "原圖數量",
            "新圖數量",
            "增減"
        ]
    )


    for row in rows:

        change = int(
            row.get(
                "change",
                0
            )
        )

        if change > 0:
            change_text = f"+{change}"

        elif change < 0:
            change_text = str(change)

        else:
            change_text = "0"


        writer.writerow(
            [
                row.get(
                    "symbol_id",
                    ""
                ),

                row.get(
                    "equipment",
                    ""
                ),

                row.get(
                    "old_count",
                    0
                ),

                row.get(
                    "new_count",
                    0
                ),

                change_text
            ]
        )


    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                "attachment; "
                "filename=DrawDiff.csv"
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

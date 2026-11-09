
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

ALLOWED = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"}

SYMBOL_NAMES = {
    0: "住戶開關箱與弱電箱整合",
    1: "馬桶專用插座 H:30cm",
    2: "抽油煙機插座 H:天花板內",
    3: "雙聯插座 H:30cm／電視牆 H:80cm",
    4: "雙聯插座 H:110cm",
    5: "洗碗機專用插座 H:45cm",
    6: "微波烤箱專用插座 220V(H:55cm)",
    7: "洗衣機用插座（防雨型）H:120cm",
    8: "廚房橫櫃燈用電源出線口 H:170cm",
    9: "暖風乾燥機電源出口 220V(H:天花)",
    10: "雙聯紅色緊急插座（弱電箱、箱體內）",
    11: "雙聯紅色緊急插座（冰箱／電視牆）",
    12: "冷氣電源出口 220V(H:天花板)"
}

MATCH_THRESHOLD = 0.64

SCALES = (
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.90, 1.00, 1.10, 1.20,
    1.35, 1.50, 1.70, 2.00
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024


# ============================================================
# 圖片讀取
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
        doc = fitz.open(stream=raw, filetype="pdf")
        try:
            if len(doc) == 0:
                return None

            page = doc.load_page(0)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2.0, 2.0),
                alpha=False
            )
            png_bytes = pix.tobytes("png")
        finally:
            doc.close()

        arr = np.frombuffer(png_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def image_to_base64(image):
    if image is None:
        return None

    ok, buffer = cv2.imencode(".png", image)

    if not ok:
        return None

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return "data:image/png;base64," + encoded


# ============================================================
# 裁除圖面白邊
# ============================================================

def crop_white_border(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 245

    coords = np.column_stack(np.where(mask))

    if len(coords) < 100:
        return image

    y, x, h, w = cv2.boundingRect(
        coords.astype(np.int32)
    )

    padding = 15

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    return image[y1:y2, x1:x2]


# ============================================================
# 清理品項模板
# ============================================================

def clean_symbol_template(image):
    """
    清除圖例表格邊框，只留下真正的設備符號。

    你的模板右側有垂直表格線，
    這裡會自動辨識並裁除。
    """

    gray = image.copy()

    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    ink = gray < 225

    # 找右側是否有貫穿整張模板的直線
    right_start = int(w * 0.75)

    for x in range(w - 1, right_start - 1, -1):
        coverage = np.mean(ink[:, x])

        if coverage > 0.70:
            gray = gray[:, :max(1, x - 1)]
            break

    # 找上下表格橫線
    h, w = gray.shape
    ink = gray < 225

    if np.mean(ink[0, :]) > 0.70:
        gray = gray[1:, :]

    h, w = gray.shape
    ink = gray < 225

    if np.mean(ink[-1, :]) > 0.70:
        gray = gray[:-1, :]

    # 再裁除符號以外的白色區域
    mask = gray < 235
    coords = np.column_stack(np.where(mask))

    if len(coords) < 8:
        return None

    y, x, h, w = cv2.boundingRect(
        coords.astype(np.int32)
    )

    padding = 2

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(gray.shape[1], x + w + padding)
    y2 = min(gray.shape[0], y + h + padding)

    result = gray[y1:y2, x1:x2]

    if result.shape[0] < 8 or result.shape[1] < 8:
        return None

    return result


# ============================================================
# 載入 13 種品項圖案
# ============================================================

def load_symbol_templates():
    templates = {}
    missing = []

    for symbol_id in range(13):
        path = SYMBOL_DIR / f"{symbol_id}.png"

        if not path.exists():
            missing.append(f"{symbol_id}.png")
            continue

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            missing.append(f"{symbol_id}.png")
            continue

        cleaned = clean_symbol_template(image)

        if cleaned is None:
            missing.append(f"{symbol_id}.png")
            continue

        templates[symbol_id] = cleaned

    return templates, missing


# ============================================================
# 圖面比對前處理
# ============================================================

def prepare_plan(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 降低掃描和 JPEG 雜訊
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


# ============================================================
# 計算兩個框的重疊程度
# ============================================================

def calculate_iou(a, b):
    ax1 = a["x"]
    ay1 = a["y"]
    ax2 = ax1 + a["w"]
    ay2 = ay1 + a["h"]

    bx1 = b["x"]
    by1 = b["y"]
    bx2 = bx1 + b["w"]
    by2 = by1 + b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1)
        * max(0, iy2 - iy1)
    )

    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# 單一品項的重複辨識排除
# ============================================================

def suppress_duplicates(matches, threshold=0.30):
    """
    不同縮放比例可能找到同一個符號多次。
    只保留分數最高的辨識結果。
    """

    matches = sorted(
        matches,
        key=lambda item: item["score"],
        reverse=True
    )

    kept = []

    for candidate in matches:
        duplicate = False

        for existing in kept:
            if calculate_iou(candidate, existing) >= threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


# ============================================================
# 多尺度設備符號辨識
# ============================================================

def detect_one_symbol(plan_gray, template):
    detections = []

    template_h, template_w = template.shape

    # 圖例符號採用灰階相關係數比對
    for scale in SCALES:

        width = max(8, int(template_w * scale))
        height = max(8, int(template_h * scale))

        if width >= plan_gray.shape[1]:
            continue

        if height >= plan_gray.shape[0]:
            continue

        resized = cv2.resize(
            template,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

        if np.std(resized) < 3:
            continue

        response = cv2.matchTemplate(
            plan_gray,
            resized,
            cv2.TM_CCOEFF_NORMED
        )

        # 以局部最大值找候選位置，
        # 避免同一區域產生大量相鄰結果
        kernel_w = max(3, int(width * 0.55))
        kernel_h = max(3, int(height * 0.55))

        if kernel_w % 2 == 0:
            kernel_w += 1

        if kernel_h % 2 == 0:
            kernel_h += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_w, kernel_h)
        )

        local_max = cv2.dilate(response, kernel)

        candidate_mask = (
            (response >= MATCH_THRESHOLD)
            & (response >= local_max - 0.00001)
        )

        ys, xs = np.where(candidate_mask)

        if len(xs) > 500:
            scores = response[ys, xs]
            top_indices = np.argsort(scores)[-500:]
            ys = ys[top_indices]
            xs = xs[top_indices]

        for y, x in zip(ys, xs):
            detections.append({
                "x": int(x),
                "y": int(y),
                "w": int(width),
                "h": int(height),
                "score": float(response[y, x])
            })

    return suppress_duplicates(detections)


# ============================================================
# 13 種品項全部辨識
# ============================================================

def detect_symbols(image, templates):
    gray = prepare_plan(image)

    results = {}

    for symbol_id, template in templates.items():
        matches = detect_one_symbol(
            gray,
            template
        )

        results[symbol_id] = matches

    return results


# ============================================================
# 同一位置跨品項重複辨識排除
# ============================================================

def remove_cross_class_duplicates(detections):
    """
    如果不同品項模板在完全相同位置被辨識，
    保留相似度較高的結果，避免重複計數。
    """

    all_items = []

    for symbol_id, matches in detections.items():
        for item in matches:
            current = dict(item)
            current["symbol_id"] = symbol_id
            all_items.append(current)

    all_items.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    kept = []

    for candidate in all_items:
        duplicate = False

        for existing in kept:
            same_area = calculate_iou(
                candidate,
                existing
            )

            if same_area >= 0.55:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    cleaned = {
        symbol_id: []
        for symbol_id in detections
    }

    for item in kept:
        cleaned[item["symbol_id"]].append({
            "x": item["x"],
            "y": item["y"],
            "w": item["w"],
            "h": item["h"],
            "score": item["score"]
        })

    return cleaned


# ============================================================
# 自動對齊
# ============================================================

def align_images(old_image, new_image):
    old_gray = prepare_plan(old_image)
    new_gray = prepare_plan(new_image)

    orb = cv2.ORB_create(
        nfeatures=10000
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
            (old_image.shape[1], old_image.shape[0]),
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
            (old_image.shape[1], old_image.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

    good = sorted(
        good,
        key=lambda match: match.distance
    )[:2500]

    src = np.float32([
        kp2[m.queryIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    dst = np.float32([
        kp1[m.trainIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=10000,
        confidence=0.995
    )

    if matrix is None:
        return cv2.resize(
            new_image,
            (old_image.shape[1], old_image.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

    if inliers is not None:
        inlier_ratio = float(np.mean(inliers))

        if inlier_ratio < 0.15:
            return cv2.resize(
                new_image,
                (old_image.shape[1], old_image.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )

    return cv2.warpAffine(
        new_image,
        matrix,
        (old_image.shape[1], old_image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


# ============================================================
# 在圖面上標記辨識出的設備
# ============================================================

def draw_detected_symbols(image, detections):
    result = image.copy()

    for symbol_id, matches in detections.items():
        for match in matches:
            x = match["x"]
            y = match["y"]
            w = match["w"]
            h = match["h"]

            center_x = x + w // 2
            center_y = y + h // 2

            radius = max(
                10,
                int(max(w, h) * 0.60)
            )

            cv2.circle(
                result,
                (center_x, center_y),
                radius,
                (0, 0, 255),
                2
            )

            label = str(symbol_id)

            label_x = max(
                12,
                center_x - radius
            )

            label_y = max(
                12,
                center_y - radius
            )

            cv2.circle(
                result,
                (label_x, label_y),
                12,
                (0, 0, 255),
                -1
            )

            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                1
            )

            text_x = int(
                label_x - text_size[0] / 2
            )

            text_y = int(
                label_y + text_size[1] / 2
            )

            cv2.putText(
                result,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return result


# ============================================================
# 首頁
# ============================================================

@app.get("/")
def index():
    return render_template("index.html")


# ============================================================
# 數量辨識 API
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

    try:
        old_image = read_upload(old_file)
        new_image = read_upload(new_file)
    except Exception as error:
        return jsonify(
            error=f"圖片讀取失敗：{str(error)}"
        ), 400

    if old_image is None or new_image is None:
        return jsonify(
            error="圖片無法讀取"
        ), 400

    old_image = crop_white_border(old_image)
    new_image = crop_white_border(new_image)

    # 1. 先對齊
    aligned_new = align_images(
        old_image,
        new_image
    )

    # 2. 讀取 13 張品項圖案
    templates, missing = load_symbol_templates()

    if missing:
        return jsonify(
            error="以下品項圖案不存在或無法讀取：" + "、".join(missing)
        ), 500

    # 3. 分別辨識原圖和修改後圖
    old_detections = detect_symbols(
        old_image,
        templates
    )

    new_detections = detect_symbols(
        aligned_new,
        templates
    )

    # 4. 排除跨品項重複
    old_detections = remove_cross_class_duplicates(
        old_detections
    )

    new_detections = remove_cross_class_duplicates(
        new_detections
    )

    # 5. 統計每種設備數量
    rows = []

    added_total = 0
    removed_total = 0
    changed_types = 0

    for symbol_id in range(13):
        old_count = len(
            old_detections.get(symbol_id, [])
        )

        new_count = len(
            new_detections.get(symbol_id, [])
        )

        difference = new_count - old_count

        if difference > 0:
            added_total += difference

        elif difference < 0:
            removed_total += abs(difference)

        if difference != 0:
            changed_types += 1

        rows.append({
            "id": symbol_id,
            "name": SYMBOL_NAMES[symbol_id],
            "old_count": old_count,
            "new_count": new_count,
            "difference": difference
        })

    # 6. 圖面標記
    old_marked = draw_detected_symbols(
        old_image,
        old_detections
    )

    new_marked = draw_detected_symbols(
        aligned_new,
        new_detections
    )

    return jsonify(
        difference_count=changed_types,
        added_count=added_total,
        removed_count=removed_total,
        old_image=image_to_base64(old_marked),
        new_image=image_to_base64(new_marked),
        rows=rows
    )


# ============================================================
# CSV 匯出
# ============================================================

@app.post("/export")
def export_csv():
    data = request.get_json() or {}
    rows = data.get("rows", [])

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "品項編號",
        "品項名稱",
        "原始圖數量",
        "修改後圖數量",
        "增減"
    ])

    for row in rows:
        writer.writerow([
            row.get("id", ""),
            row.get("name", ""),
            row.get("old_count", 0),
            row.get("new_count", 0),
            row.get("difference", 0)
        ])

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                "attachment; filename=DrawDiff設備數量.csv"
        }
    )


# ============================================================
# 啟動
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )

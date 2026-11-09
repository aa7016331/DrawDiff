
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
    0.50,
    0.65,
    0.80,
    0.95,
    1.10,
    1.30
)

MAX_PLAN_SIDE = 1200
MAX_PEAKS_PER_SCALE = 20


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

    matches = sorted(
        matches,
        key=lambda item: item["score"],
        reverse=True
    )

    kept = []

    for candidate in matches:

        duplicate = False

        cx = candidate["x"] + candidate["w"] / 2
        cy = candidate["y"] + candidate["h"] / 2

        for existing in kept:

            ex = existing["x"] + existing["w"] / 2
            ey = existing["y"] + existing["h"] / 2

            max_size = max(
                candidate["w"],
                candidate["h"],
                existing["w"],
                existing["h"]
            )

            distance = (
                (cx - ex) ** 2 +
                (cy - ey) ** 2
            ) ** 0.5

            if (
                calculate_iou(candidate, existing) >= threshold
                or distance <= max_size * 0.30
            ):
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

    if template_h < 5 or template_w < 5:
        return detections

    for scale in SCALES:

        width = max(5, int(template_w * scale))
        height = max(5, int(template_h * scale))

        if width >= plan_gray.shape[1]:
            continue

        if height >= plan_gray.shape[0]:
            continue

        interpolation = (
            cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_CUBIC
        )

        resized = cv2.resize(
            template,
            (width, height),
            interpolation=interpolation
        )

        if np.std(resized) < 3:
            continue

        response = cv2.matchTemplate(
            plan_gray,
            resized,
            cv2.TM_CCOEFF_NORMED
        )

        for _ in range(MAX_PEAKS_PER_SCALE):

            _, max_score, _, max_location = cv2.minMaxLoc(
                response
            )

            if max_score < MATCH_THRESHOLD:
                break

            x, y = max_location

            detections.append({
                "x": int(x),
                "y": int(y),
                "w": int(width),
                "h": int(height),
                "score": float(max_score)
            })

            half_w = max(3, width // 2)
            half_h = max(3, height // 2)

            x1 = max(0, x - half_w)
            y1 = max(0, y - half_h)

            x2 = min(
                response.shape[1],
                x + half_w + 1
            )

            y2 = min(
                response.shape[0],
                y + half_h + 1
            )

            response[y1:y2, x1:x2] = -1

    return suppress_duplicates(detections)



# ============================================================
# 13 種品項全部辨識
# ============================================================


def detect_symbols(image, templates):

    gray = prepare_plan(image)

    original_h, original_w = gray.shape

    # 大圖才縮小，保留小型設備符號的細節
    ratio = min(
        1.0,
        1600 / max(original_h, original_w)
    )

    if ratio < 1.0:
        search_w = max(1, int(original_w * ratio))
        search_h = max(1, int(original_h * ratio))

        gray = cv2.resize(
            gray,
            (search_w, search_h),
            interpolation=cv2.INTER_AREA
        )

    scale_x = original_w / gray.shape[1]
    scale_y = original_h / gray.shape[0]

    edge = cv2.Canny(gray, 45, 130)

    # 依照實際圖面設定各品項的比對門檻
    thresholds = {
        0: 0.72,
        1: 0.53,
        2: 0.50,
        3: 0.65,
        4: 0.58,
        5: 0.58,
        6: 0.75,
        7: 0.60,
        8: 0.54,
        9: 0.76,
        10: 0.73,
        11: 0.73,
        12: 0.60
    }

    # 不同品項使用不同最小縮放比例
    min_scales = {
        0: 0.40,
        1: 0.35,
        2: 0.35,
        3: 0.40,
        4: 0.40,
        5: 0.40,
        6: 0.45,
        7: 0.40,
        8: 0.40,
        9: 0.50,
        10: 0.50,
        11: 0.50,
        12: 0.55
    }

    letter_ids = {
        2, 4, 5, 6, 7, 9, 10, 11, 12
    }

    letter_thresholds = {
        2: 0.35,
        4: 0.40,
        5: 0.40,
        6: 0.72,
        7: 0.42,
        9: 0.56,
        10: 0.42,
        11: 0.50,
        12: 0.56
    }

    core_x_fraction = {
        0: 0.50,
        1: 0.50,
        2: 0.40,
        3: 0.50,
        4: 0.42,
        5: 0.40,
        6: 0.40,
        7: 0.40,
        8: 0.50,
        9: 0.40,
        10: 0.40,
        11: 0.40,
        12: 0.40
    }

    # --------------------------------------------------------
    # 旋轉模板
    # --------------------------------------------------------

    def rotate_image(img, angle):

        if angle == 0:
            return img

        if angle == 90:
            return cv2.rotate(
                img,
                cv2.ROTATE_90_CLOCKWISE
            )

        if angle == 180:
            return cv2.rotate(
                img,
                cv2.ROTATE_180
            )

        return cv2.rotate(
            img,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    def rotate_point(x, y, w, h, angle):

        if angle == 0:
            return x, y

        if angle == 90:
            return h - 1 - y, x

        if angle == 180:
            return w - 1 - x, h - 1 - y

        return y, w - 1 - x

    # --------------------------------------------------------
    # 驗證外圈是否有真正的圓形線條
    # --------------------------------------------------------

    def check_circle(edge_image, cx, cy, radius):

        if radius < 3:
            return 0.0

        sectors = 36
        matched = 0

        for i in range(sectors):

            angle = (
                2 * np.pi * (i + 0.5) / sectors
            )

            found = False

            for r in (
                radius - 1.5,
                radius,
                radius + 1.5
            ):

                px = int(round(
                    cx + r * np.cos(angle)
                ))

                py = int(round(
                    cy + r * np.sin(angle)
                ))

                if (
                    0 <= px < edge_image.shape[1]
                    and 0 <= py < edge_image.shape[0]
                    and edge_image[py, px] > 0
                ):

                    found = True
                    break

            if found:
                matched += 1

        return matched / sectors

    # --------------------------------------------------------
    # 驗證符號旁邊的英文字母
    # --------------------------------------------------------

    def check_letter(
        original_template,
        resized_template,
        mask,
        x,
        y
    ):

        mask_resized = cv2.resize(
            mask,
            (
                resized_template.shape[1],
                resized_template.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

        ys, xs = np.where(mask_resized > 0)

        if len(xs) < 5:
            return None

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1

        letter_template = resized_template[
            y1:y2,
            x1:x2
        ]

        if (
            letter_template.size == 0
            or np.std(letter_template) < 3
        ):
            return None

        best_score = -1.0

        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):

                tx = x + x1 + dx
                ty = y + y1 + dy

                if (
                    tx < 0
                    or ty < 0
                    or tx + letter_template.shape[1] > gray.shape[1]
                    or ty + letter_template.shape[0] > gray.shape[0]
                ):
                    continue

                patch = gray[
                    ty:ty + letter_template.shape[0],
                    tx:tx + letter_template.shape[1]
                ]

                if np.std(patch) < 3:
                    continue

                score = cv2.matchTemplate(
                    patch,
                    letter_template,
                    cv2.TM_CCOEFF_NORMED
                )[0, 0]

                best_score = max(
                    best_score,
                    float(score)
                )

        return (
            best_score
            if best_score >= 0
            else None
        )

    # --------------------------------------------------------
    # 開始辨識
    # --------------------------------------------------------

    results = {}

    scales = (
        0.35,
        0.42,
        0.49,
        0.56,
        0.63,
        0.70,
        0.77,
        0.84
    )

    for symbol_id, template in templates.items():

        candidates = []

        th0, tw0 = template.shape

        threshold = thresholds.get(
            symbol_id,
            0.70
        )

        # ----------------------------------------------------
        # 四個方向
        # ----------------------------------------------------

        for angle in (
            0,
            90,
            180,
            270
        ):

            rotated = rotate_image(
                template,
                angle
            )

            rh, rw = rotated.shape

            core_x, core_y = rotate_point(
                tw0 * core_x_fraction[symbol_id],
                th0 * 0.5,
                tw0,
                th0,
                angle
            )

            # ------------------------------------------------
            # 多尺寸搜尋
            # ------------------------------------------------

            for scale in scales:

                if scale < min_scales[symbol_id]:
                    continue

                width = max(
                    8,
                    int(rw * scale)
                )

                height = max(
                    8,
                    int(rh * scale)
                )

                if (
                    width >= gray.shape[1]
                    or height >= gray.shape[0]
                ):
                    continue

                resized = cv2.resize(
                    rotated,
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )

                response = cv2.matchTemplate(
                    gray,
                    resized,
                    cv2.TM_CCOEFF_NORMED
                )

                # 只取局部最高相似度位置
                kw = max(
                    3,
                    int(width * 0.60) // 2 * 2 + 1
                )

                kh = max(
                    3,
                    int(height * 0.60) // 2 * 2 + 1
                )

                local_max = cv2.dilate(
                    response,
                    cv2.getStructuringElement(
                        cv2.MORPH_RECT,
                        (kw, kh)
                    )
                )

                ys, xs = np.where(
                    (response >= threshold)
                    &
                    (response >= local_max - 0.000001)
                )

                # 限制每種尺寸的候選數量
                if len(xs) > 15:

                    scores = response[ys, xs]

                    indices = np.argsort(
                        scores
                    )[-15:]

                    ys = ys[indices]
                    xs = xs[indices]

                for yy, xx in zip(ys, xs):

                    score = float(
                        response[yy, xx]
                    )

                    # ----------------------------------------
                    # 外圈驗證
                    # ----------------------------------------

                    if symbol_id != 0:

                        cx = (
                            xx
                            + core_x * width / rw
                        )

                        cy = (
                            yy
                            + core_y * height / rh
                        )

                        radius = (
                            min(th0, tw0)
                            * 0.36
                            * scale
                        )

                        circle_score = check_circle(
                            edge,
                            cx,
                            cy,
                            radius
                        )

                        if circle_score < 0.48:
                            continue

                    else:

                        circle_score = 1.0

                    # ----------------------------------------
                    # 字母驗證
                    # ----------------------------------------

                    letter_score = None

                    if symbol_id in letter_ids:

                        letter_mask = np.zeros(
                            (th0, tw0),
                            np.uint8
                        )

                        letter_mask[
                            int(th0 * 0.38):,
                            int(tw0 * 0.68):
                        ] = 255

                        rotated_mask = rotate_image(
                            letter_mask,
                            angle
                        )

                        letter_score = check_letter(
                            rotated,
                            resized,
                            rotated_mask,
                            int(xx),
                            int(yy)
                        )

                        if (
                            letter_score is None
                            or letter_score < letter_thresholds[symbol_id]
                        ):
                            continue

                    # ----------------------------------------
                    # 綜合分數
                    # ----------------------------------------

                    combined_score = score

                    if letter_score is not None:

                        combined_score = (
                            0.70 * score
                            +
                            0.30 * letter_score
                        )

                    combined_score = (
                        0.90 * combined_score
                        +
                        0.10 * circle_score
                    )

                    candidates.append({
                        "x": int(xx),
                        "y": int(yy),
                        "w": int(width),
                        "h": int(height),
                        "score": float(combined_score),
                        "angle": angle
                    })

        # ----------------------------------------------------
        # 同一品項重複結果排除
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        kept = []

        for candidate in candidates:

            cx = (
                candidate["x"]
                + candidate["w"] / 2
            )

            cy = (
                candidate["y"]
                + candidate["h"] / 2
            )

            duplicate = False

            for existing in kept:

                ex = (
                    existing["x"]
                    + existing["w"] / 2
                )

                ey = (
                    existing["y"]
                    + existing["h"] / 2
                )

                distance = (
                    (cx - ex) ** 2
                    +
                    (cy - ey) ** 2
                ) ** 0.5

                min_size = min(
                    candidate["w"],
                    candidate["h"],
                    existing["w"],
                    existing["h"]
                )

                if distance < max(
                    7,
                    min_size * 0.65
                ):

                    duplicate = True
                    break

            if not duplicate:
                kept.append(candidate)

        # ----------------------------------------------------
        # 還原原始圖片座標
        # ----------------------------------------------------

        restored = []

        for item in kept:

            restored.append({
                "x": int(round(item["x"] * scale_x)),
                "y": int(round(item["y"] * scale_y)),
                "w": int(round(item["w"] * scale_x)),
                "h": int(round(item["h"] * scale_y)),
                "score": item["score"]
            })

        results[symbol_id] = restored

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

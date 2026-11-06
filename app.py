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
        (5

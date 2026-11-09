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
# 品項圖案資料夾
# ============================================================

SYMBOL_DIR = (
    BASE
    / "static"
    / "symbols"
)


# ============================================================
# 圖案名稱
#
# 請依照你 0.png ~ 12.png 的實際對應順序修改名稱
# ============================================================

SYMBOL_NAMES = {

    0: "住宅開關箱與弱電箱整合",

    1: "馬桶專用插座 H:30cm",

    2: "抽油煙機插座 H:天花板內",

    3: "雙聯插座 H:30cm，電視牆 H:80cm",

    4: "雙聯插座 H:110cm",

    5: "洗碗機專用插座 H:45cm",

    6: "微波烤箱專用插座 220V",

    7: "洗衣機專用插座 H:120cm",

    8: "廚房檯櫃用電源出線口 H:170cm",

    9: "暖風乾燥機電源出口 220V",

    10: "雙聯紅色緊急插座",

    11: "雙聯紅色緊急插座（冰箱 H:30cm）",

    12: "冷氣電源出口 220V"
}


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    40 * 1024 * 1024
)


# ============================================================
# 基本工具
# ============================================================

def allowed(filename):

    return (

        "."
        in filename

        and

        filename.rsplit(
            ".",
            1
        )[1].lower()

        in ALLOWED
    )


def read_upload(file):

    ext = file.filename.rsplit(
        ".",
        1
    )[1].lower()

    raw = file.read()


    if ext == "pdf":

        doc = fitz.open(
            stream=raw,
            filetype="pdf"
        )


        page = doc.load_page(
            0
        )


        pix = page.get_pixmap(

            matrix=fitz.Matrix(
                2.0,
                2.0
            ),

            alpha=False
        )


        png_bytes = pix.tobytes(
            "png"
        )


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

    ).decode(
        "utf-8"
    )


    return (

        "data:image/png;base64,"
        + encoded

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
        coords.astype(
            np.int32
        )
    )


    padding = 15


    x1 = max(
        0,
        x - padding
    )


    y1 = max(
        0,
        y - padding
    )


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
# 載入品項圖案
# ============================================================

def load_symbol_templates():

    templates = {}


    for symbol_id in range(
        13
    ):


        path = (
            SYMBOL_DIR
            / f"{symbol_id}.png"
        )


        if not path.exists():

            print(
                f"找不到品項圖案：{path}"
            )

            continue


        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            continue


        # 去掉圖案周圍白邊
        mask = image < 245


        coords = np.column_stack(
            np.where(mask)
        )


        if len(coords) > 10:

            y, x, h, w = cv2.boundingRect(
                coords.astype(
                    np.int32
                )
            )


            padding = 2


            x1 = max(
                0,
                x - padding
            )

            y1 = max(
                0,
                y - padding
            )

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


        templates[
            symbol_id
        ] = image


    return templates


# ============================================================
# 多尺度圖案辨識
# ============================================================

def detect_symbols(
    image,
    templates
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    results = {}


    # ========================================================
    # 每一個品項圖案
    # ========================================================

    for symbol_id, template in templates.items():


        matches = []


        template_h, template_w = (
            template.shape[:2]
        )


        # ----------------------------------------------------
        # 不同縮放比例
        #
        # 因為 PDF、圖片解析度不同，
        # 圖面上的符號大小可能不同
        # ----------------------------------------------------

        scales = [

            0.45,
            0.55,
            0.65,
            0.75,
            0.85,

            0.95,
            1.00,
            1.10,
            1.20,
            1.35,
            1.50,
            1.70,
            2.00

        ]


        for scale in scales:


            width = int(
                template_w
                * scale
            )


            height = int(
                template_h
                * scale
            )


            if (
                width < 8
                or height < 8
            ):

                continue


            if (
                width >= gray.shape[1]
                or height >= gray.shape[0]
            ):

                continue


            resized_template = cv2.resize(

                template,

                (
                    width,
                    height
                ),

                interpolation=cv2.INTER_CUBIC

            )


            # ------------------------------------------------
            # Template Matching
            # ------------------------------------------------

            response = cv2.matchTemplate(

                gray,

                resized_template,

                cv2.TM_CCOEFF_NORMED

            )


            # ------------------------------------------------
            # 相似度門檻
            #
            # 先設定偏高，避免大量誤判
            # ------------------------------------------------

            threshold = 0.72


            locations = np.where(
                response >= threshold
            )


            for y, x in zip(
                locations[0],
                locations[1]
            ):


                score = float(
                    response[y, x]
                )


                matches.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(width),
                        "h": int(height),
                        "score": score
                    }
                )


        # ====================================================
        # Non-Maximum Suppression
        #
        # 同一個符號通常會被不同尺度重複抓到，
        # 這裡把重複框刪掉
        # ====================================================

        matches = sorted(

            matches,

            key=lambda item:
                item["score"],

            reverse=True

        )


        final_matches = []


        for match in matches:


            mx1 = match["x"]

            my1 = match["y"]

            mx2 = mx1 + match["w"]

            my2 = my1 + match["h"]


            duplicate = False


            for existing in final_matches:


                ex1 = existing["x"]

                ey1 = existing["y"]

                ex2 = ex1 + existing["w"]

                ey2 = ey1 + existing["h"]


                ix1 = max(
                    mx1,
                    ex1
                )

                iy1 = max(
                    my1,
                    ey1
                )

                ix2 = min(
                    mx2,
                    ex2
                )

                iy2 = min(
                    my2,
                    ey2
                )


                intersection = max(
                    0,
                    ix2 - ix1
                ) * max(
                    0,
                    iy2 - iy1
                )


                area_match = (
                    match["w"]
                    * match["h"]
                )


                area_existing = (
                    existing["w"]
                    * existing["h"]
                )


                union = (

                    area_match

                    +

                    area_existing

                    -

                    intersection

                )


                iou = (

                    intersection
                    / union

                    if union > 0

                    else 0

                )


                # 同位置高度重疊就視為同一個
                if iou > 0.30:

                    duplicate = True

                    break


            if not duplicate:

                final_matches.append(
                    match
                )


        results[
            symbol_id
        ] = final_matches


    return results


# ============================================================
# 畫出辨識結果
# ============================================================

def draw_detected_symbols(
    image,
    detections
):

    result = image.copy()


    for symbol_id, matches in detections.items():


        for index, match in enumerate(
            matches,
            start=1
        ):


            x = match["x"]

            y = match["y"]

            w = match["w"]

            h = match["h"]


            # =================================================
            # 紅色圈
            # =================================================

            center_x = x + w // 2

            center_y = y + h // 2


            radius = max(

                12,

                int(
                    max(
                        w,
                        h
                    )
                    * 0.70
                )

            )


            cv2.circle(

                result,

                (
                    center_x,
                    center_y
                ),

                radius,

                (
                    0,
                    0,
                    255
                ),

                2

            )


            # =================================================
            # 品項編號
            # =================================================

            label = str(
                symbol_id
            )


            label_x = max(
                10,
                center_x - radius
            )


            label_y = max(
                12,
                center_y - radius
            )


            cv2.circle(

                result,

                (
                    label_x,
                    label_y
                ),

                12,

                (
                    0,
                    0,
                    255
                ),

                -1

            )


            cv2.putText(

                result,

                label,

                (
                    label_x - 5,
                    label_y + 5
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.45,

                (
                    255,
                    255,
                    255
                ),

                1,

                cv2.LINE_AA

            )


    return result


# ============================================================
# 自動對齊
# ============================================================

def preprocess_for_alignment(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    return cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )


def align_images(
    old_image,
    new_image
):

    old_gray = preprocess_for_alignment(
        old_image
    )


    new_gray = preprocess_for_alignment(
        new_image
    )


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

            (
                old_image.shape[1],
                old_image.shape[0]
            )

        )


    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
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

            good.append(
                m
            )


    if len(good) < 10:

        return cv2.resize(

            new_image,

            (
                old_image.shape[1],
                old_image.shape[0]
            )

        )


    src = np.float32(
        [
            kp2[m.queryIdx].pt
            for m in good
        ]
    ).reshape(
        -1,
        1,
        2
    )


    dst = np.float32(
        [
            kp1[m.trainIdx].pt
            for m in good
        ]
    ).reshape(
        -1,
        1,
        2
    )


    matrix, _ = cv2.estimateAffinePartial2D(

        src,

        dst,

        method=cv2.RANSAC,

        ransacReprojThreshold=2.5

    )


    if matrix is None:

        return cv2.resize(

            new_image,

            (
                old_image.shape[1],
                old_image.shape[0]
            )

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

        borderValue=(
            255,
            255,
            255
        )

    )


# ============================================================
# 首頁
# ============================================================

@app.get("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# 分析
# ============================================================

@app.post("/analyze")
def analyze_api():


    old_file = request.files.get(
        "old"
    )


    new_file = request.files.get(
        "new"
    )


    if (
        not old_file
        or not new_file
    ):

        return jsonify(

            error="請同時上傳原始圖與修改後圖"

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


    # ========================================================
    # 1. 裁掉白邊
    # ========================================================

    old_image = crop_white_border(
        old_image
    )


    new_image = crop_white_border(
        new_image
    )


    # ========================================================
    # 2. 對齊修改後圖
    # ========================================================

    aligned_new = align_images(

        old_image,

        new_image

    )


    # ========================================================
    # 3. 載入 0～12 品項圖案
    # ========================================================

    templates = load_symbol_templates()


    if not templates:

        return jsonify(

            error=(
                "找不到品項圖案，請確認 "
                "static/symbols/0.png ~ 12.png"
            )

        ), 500


    # ========================================================
    # 4. 辨識原始圖
    # ========================================================

    old_detections = detect_symbols(

        old_image,

        templates

    )


    # ========================================================
    # 5. 辨識修改後圖
    # ========================================================

    new_detections = detect_symbols(

        aligned_new,

        templates

    )


    # ========================================================
    # 6. 圖面畫圈
    # ========================================================

    old_marked = draw_detected_symbols(

        old_image,

        old_detections

    )


    new_marked = draw_detected_symbols(

        aligned_new,

        new_detections

    )


    # ========================================================
    # 7. 建立品項統計
    # ========================================================

    rows = []


    for symbol_id in range(
        13
    ):


        old_count = len(

            old_detections.get(
                symbol_id,
                []
            )

        )


        new_count = len(

            new_detections.get(
                symbol_id,
                []
            )

        )


        difference = (

            new_count

            -

            old_count

        )


        rows.append(

            {

                "id":
                    symbol_id,

                "name":
                    SYMBOL_NAMES.get(
                        symbol_id,
                        f"品項 {symbol_id}"
                    ),

                "old_count":
                    old_count,

                "new_count":
                    new_count,

                "difference":
                    difference

            }

        )


    # ========================================================
    # 有變化的品項數量
    # ========================================================

    difference_count = len(

        [
            row

            for row in rows

            if row["difference"] != 0
        ]

    )


    return jsonify(

        difference_count=
            difference_count,

        old_image=
            image_to_base64(
                old_marked
            ),

        new_image=
            image_to_base64(
                new_marked
            ),

        rows=
            rows

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
            "品項編號",
            "品項名稱",
            "原始圖數量",
            "修改後數量",
            "增減"
        ]
    )


    for row in rows:


        writer.writerow(

            [

                row.get(
                    "id",
                    ""
                ),

                row.get(
                    "name",
                    ""
                ),

                row.get(
                    "old_count",
                    ""
                ),

                row.get(
                    "new_count",
                    ""
                ),

                row.get(
                    "difference",
                    ""
                )

            ]

        )


    return Response(

        "\ufeff"
        + output.getvalue(),

        mimetype=
            "text/csv; charset=utf-8",

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
```

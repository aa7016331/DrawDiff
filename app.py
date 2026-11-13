import os
import base64
from pathlib import Path

import cv2
import fitz
import numpy as np

from flask import Flask, request, render_template_string


# ============================================================
# 基本設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SYMBOL_DIR = BASE_DIR / "static" / "symbols"

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "pdf"
}

app = Flask(__name__)


# ============================================================
# 網頁 HTML
# ============================================================

PAGE_HTML = """
<!DOCTYPE html>
<html lang="zh-Hant">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <title>DrawDiff 圖面設備比對</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 30px;

            font-family:
                Arial,
                "Microsoft JhengHei",
                sans-serif;

            background: #f3f6f8;
            color: #1f2933;
        }

        .container {
            max-width: 1500px;
            margin: auto;
        }

        h1 {
            margin-bottom: 10px;
        }

        .card {
            background: white;
            border: 1px solid #d9e2e8;
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
        }

        .upload-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .upload-box {
            border: 2px dashed #b9c9d3;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            background: #fafcfd;
        }

        .upload-box h3 {
            margin-top: 0;
        }

        input[type=file] {
            max-width: 100%;
            margin-bottom: 15px;
        }

        .page-select {
            margin-top: 12px;
            padding: 14px;
            background: #edf4f6;
            border-radius: 8px;
            text-align: left;
        }

        .page-select label {
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
        }

        input[type=number] {
            width: 100%;
            padding: 10px;
            border: 1px solid #b9c9d3;
            border-radius: 6px;
            font-size: 16px;
        }

        .page-help {
            color: #60727c;
            font-size: 13px;
            margin-top: 8px;
            line-height: 1.5;
        }

        .button-row {
            text-align: center;
            margin-top: 20px;
        }

        button {
            border: 0;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            background: #16857a;
            color: white;
        }

        button:hover {
            background: #116d64;
        }

        .summary {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 20px;
        }

        .badge {
            background: #edf4f6;
            padding: 10px 18px;
            border-radius: 30px;
            font-weight: bold;
        }

        .add {
            color: #0b7a3d;
        }

        .remove {
            color: #c62828;
        }

        .image-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .image-card {
            border: 1px solid #cfdce3;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            overflow: hidden;
        }

        .image-card h3 {
            margin-top: 0;
        }

        .image-card img {
            width: 100%;
            max-height: 900px;
            object-fit: contain;
            border: 1px solid #dbe4e8;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 25px;
        }

        th {
            background: #e8f0f3;
        }

        th,
        td {
            padding: 12px;
            border-bottom: 1px solid #dbe4e8;
            text-align: center;
        }

        .error {
            padding: 15px;
            border-radius: 8px;
            background: #ffe8e8;
            color: #b71c1c;
            margin-bottom: 20px;
        }

        .small {
            color: #60727c;
            font-size: 14px;
            margin-top: 8px;
            line-height: 1.6;
        }

        @media (max-width: 900px) {

            body {
                padding: 12px;
            }

            .upload-grid,
            .image-grid {
                grid-template-columns: 1fr;
            }

        }

    </style>

</head>


<body>

<div class="container">

    <h1>DrawDiff 圖面設備辨識</h1>


    <div class="card">

        <form
            method="POST"
            action="/analyze"
            enctype="multipart/form-data"
        >

            <div class="upload-grid">


                <!-- ====================================== -->
                <!-- 原始圖 -->
                <!-- ====================================== -->

                <div class="upload-box">

                    <h3>原始圖</h3>

                    <p>JPG / PNG / PDF</p>

                    <input
                        type="file"
                        name="original_file"
                        accept=".jpg,.jpeg,.png,.pdf"
                        required
                    >


                    <div class="page-select">

                        <label for="original_page">
                            原始圖 PDF 頁數
                        </label>

                        <input
                            type="number"
                            id="original_page"
                            name="original_page"
                            value="1"
                            min="1"
                        >

                        <div class="page-help">

                            如果上傳的是 PDF，
                            請輸入要辨識的頁數。

                            <br>

                            例如：第 4 頁請輸入 4。

                            <br>

                            JPG / PNG 可維持 1。

                        </div>

                    </div>

                </div>


                <!-- ====================================== -->
                <!-- 修改後圖 -->
                <!-- ====================================== -->

                <div class="upload-box">

                    <h3>修改後圖</h3>

                    <p>JPG / PNG / PDF</p>

                    <input
                        type="file"
                        name="modified_file"
                        accept=".jpg,.jpeg,.png,.pdf"
                        required
                    >


                    <div class="page-select">

                        <label for="modified_page">
                            修改後圖 PDF 頁數
                        </label>

                        <input
                            type="number"
                            id="modified_page"
                            name="modified_page"
                            value="1"
                            min="1"
                        >

                        <div class="page-help">

                            如果上傳的是 PDF，
                            請輸入要辨識的頁數。

                            <br>

                            例如：第 3 頁請輸入 3。

                            <br>

                            JPG / PNG 可維持 1。

                        </div>

                    </div>

                </div>


            </div>


            <div class="button-row">

                <button type="submit">
                    開始自動辨識
                </button>

            </div>

        </form>

    </div>


    {% if error %}

        <div class="error">
            {{ error }}
        </div>

    {% endif %}


    {% if result %}

        <div class="card">

            <h2>插座增減結果</h2>


            <div class="summary">

                <div class="badge">
                    原圖 {{ result.original_total }} 個
                </div>

                <div class="badge">
                    修改後 {{ result.modified_total }} 個
                </div>

                <div class="badge add">
                    增加 {{ result.added }} 個
                </div>

                <div class="badge remove">
                    減少 {{ result.removed }} 個
                </div>

            </div>


            <div class="image-grid">


                <!-- 原圖 -->

                <div class="image-card">

                    <h3>
                        原始圖
                        （紅圈為辨識到的設備符號）
                    </h3>

                    <img
                        src="data:image/png;base64,{{ result.original_image }}"
                    >

                </div>


                <!-- 修改圖 -->

                <div class="image-card">

                    <h3>
                        修改後圖
                        （紅圈為辨識到的設備符號）
                    </h3>

                    <img
                        src="data:image/png;base64,{{ result.modified_image }}"
                    >

                </div>


            </div>


            <table>

                <thead>

                    <tr>

                        <th>
                            設備項目
                        </th>

                        <th>
                            原圖數量
                        </th>

                        <th>
                            修改後數量
                        </th>

                        <th>
                            增加
                        </th>

                        <th>
                            減少
                        </th>

                        <th>
                            淨增減
                        </th>

                    </tr>

                </thead>


                <tbody>

                    <tr>

                        <td>
                            全部插座／設備符號
                        </td>

                        <td>
                            {{ result.original_total }}
                        </td>

                        <td>
                            {{ result.modified_total }}
                        </td>

                        <td class="add">
                            {{ result.added }}
                        </td>

                        <td class="remove">
                            {{ result.removed }}
                        </td>

                        <td>
                            {{ result.net_change }}
                        </td>

                    </tr>

                </tbody>

            </table>


            <p class="small">

                原圖與修改後圖會各自獨立辨識。

                <br>

                PDF 會依照你輸入的頁數讀取指定頁面。

                <br>

                本次辨識只計算圖面上的設備符號，
                不讀取右側或旁邊的統計表格。

            </p>

        </div>

    {% endif %}

</div>

</body>

</html>
"""


# ============================================================
# 檔案副檔名
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# 取得檔案副檔名
# ============================================================

def get_extension(filename):

    if not filename:
        return ""

    if "." not in filename:
        return ""

    return filename.rsplit(
        ".",
        1
    )[1].lower()


# ============================================================
# 讀取圖片 / PDF 指定頁
# ============================================================

def load_image_from_upload(
    uploaded_file,
    page_number=1
):

    if uploaded_file is None:
        return None, "沒有上傳檔案。"


    filename = uploaded_file.filename


    if not allowed_file(filename):
        return None, "檔案格式不支援。"


    raw = uploaded_file.read()


    if not raw:
        return None, "檔案內容為空。"


    extension = get_extension(filename)


    # ========================================================
    # PDF
    # ========================================================

    if extension == "pdf":

        document = None

        try:

            document = fitz.open(
                stream=raw,
                filetype="pdf"
            )


            total_pages = len(document)


            if total_pages == 0:

                return (
                    None,
                    "PDF 沒有任何頁面。"
                )


            # 網頁輸入第 1 頁
            # PyMuPDF 使用索引 0
            page_index = page_number - 1


            if page_index < 0:

                return (
                    None,
                    "頁數必須從 1 開始。"
                )


            if page_index >= total_pages:

                return (
                    None,
                    f"PDF 只有 {total_pages} 頁，"
                    f"無法讀取第 {page_number} 頁。"
                )


            page = document.load_page(
                page_index
            )


            pixmap = page.get_pixmap(

                matrix=fitz.Matrix(
                    2.5,
                    2.5
                ),

                alpha=False

            )


            png_bytes = pixmap.tobytes(
                "png"
            )


            array = np.frombuffer(
                png_bytes,
                np.uint8
            )


            image = cv2.imdecode(
                array,
                cv2.IMREAD_COLOR
            )


            if image is None:

                return (
                    None,
                    "PDF 頁面轉換圖片失敗。"
                )


            return image, None


        except Exception as error:

            return (
                None,
                f"讀取 PDF 失敗：{str(error)}"
            )


        finally:

            if document is not None:

                document.close()


    # ========================================================
    # JPG / PNG
    # ========================================================

    try:

        array = np.frombuffer(
            raw,
            np.uint8
        )


        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return (
                None,
                "圖片讀取失敗。"
            )


        return image, None


    except Exception as error:

        return (
            None,
            f"圖片讀取失敗：{str(error)}"
        )


# ============================================================
# 圖片轉 Base64
# ============================================================

def image_to_base64(image):

    success, buffer = cv2.imencode(
        ".png",
        image
    )


    if not success:
        return ""


    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


# ============================================================
# 讀取符號模板
# ============================================================

def load_symbol_templates():

    templates = []


    if not SYMBOL_DIR.exists():

        return templates


    for file_path in sorted(
        SYMBOL_DIR.iterdir()
    ):

        if not file_path.is_file():
            continue


        extension = file_path.suffix.lower()


        if extension not in {
            ".png",
            ".jpg",
            ".jpeg"
        }:

            continue


        template = cv2.imread(

            str(file_path),

            cv2.IMREAD_GRAYSCALE

        )


        if template is None:
            continue


        if (
            template.shape[0] < 5
            or
            template.shape[1] < 5
        ):

            continue


        templates.append({

            "name": file_path.stem,

            "image": template

        })


    return templates


# ============================================================
# 計算 IoU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1 = box_a["x"]
    ay1 = box_a["y"]

    ax2 = ax1 + box_a["w"]
    ay2 = ay1 + box_a["h"]


    bx1 = box_b["x"]
    by1 = box_b["y"]

    bx2 = bx1 + box_b["w"]
    by2 = by1 + box_b["h"]


    intersection_x1 = max(
        ax1,
        bx1
    )

    intersection_y1 = max(
        ay1,
        by1
    )

    intersection_x2 = min(
        ax2,
        bx2
    )

    intersection_y2 = min(
        ay2,
        by2
    )


    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )


    intersection_area = (
        intersection_width
        *
        intersection_height
    )


    area_a = (
        (ax2 - ax1)
        *
        (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1)
        *
        (by2 - by1)
    )


    union_area = (
        area_a
        +
        area_b
        -
        intersection_area
    )


    if union_area <= 0:

        return 0.0


    return (
        intersection_area
        /
        union_area
    )


# ============================================================
# 同一符號內去除重複
# ============================================================

def remove_duplicate_matches(matches):

    matches = sorted(

        matches,

        key=lambda item: item["score"],

        reverse=True

    )


    kept = []


    for candidate in matches:

        duplicate = False


        for existing in kept:

            overlap = calculate_iou(
                candidate,
                existing
            )


            candidate_cx = (
                candidate["x"]
                +
                candidate["w"] / 2
            )

            candidate_cy = (
                candidate["y"]
                +
                candidate["h"] / 2
            )


            existing_cx = (
                existing["x"]
                +
                existing["w"] / 2
            )

            existing_cy = (
                existing["y"]
                +
                existing["h"] / 2
            )


            distance = (

                (
                    candidate_cx
                    -
                    existing_cx
                ) ** 2

                +

                (
                    candidate_cy
                    -
                    existing_cy
                ) ** 2

            ) ** 0.5


            min_size = min(

                max(
                    candidate["w"],
                    candidate["h"]
                ),

                max(
                    existing["w"],
                    existing["h"]
                )

            )


            if (

                overlap >= 0.30

                or

                distance <= max(
                    5,
                    min_size * 0.30
                )

            ):

                duplicate = True

                break


        if not duplicate:

            kept.append(
                candidate
            )


    return kept


# ============================================================
# 不同符號之間去除重複
# ============================================================

def remove_cross_class_duplicates(detections):

    all_items = []


    for symbol_name, matches in detections.items():

        for item in matches:

            current = dict(item)

            current["symbol_name"] = symbol_name

            all_items.append(
                current
            )


    all_items.sort(

        key=lambda item: item["score"],

        reverse=True

    )


    kept = []


    for candidate in all_items:

        duplicate = False


        candidate_cx = (
            candidate["x"]
            +
            candidate["w"] / 2
        )

        candidate_cy = (
            candidate["y"]
            +
            candidate["h"] / 2
        )


        for existing in kept:

            existing_cx = (
                existing["x"]
                +
                existing["w"] / 2
            )

            existing_cy = (
                existing["y"]
                +
                existing["h"] / 2
            )


            distance = (

                (
                    candidate_cx
                    -
                    existing_cx
                ) ** 2

                +

                (
                    candidate_cy
                    -
                    existing_cy
                ) ** 2

            ) ** 0.5


            overlap = calculate_iou(
                candidate,
                existing
            )


            min_size = min(

                max(
                    candidate["w"],
                    candidate["h"]
                ),

                max(
                    existing["w"],
                    existing["h"]
                )

            )


            if (

                overlap >= 0.50

                and

                distance <= max(
                    5,
                    min_size * 0.40
                )

            ):

                duplicate = True

                break


        if not duplicate:

            kept.append(
                candidate
            )


    cleaned = {

        symbol_name: []

        for symbol_name
        in detections

    }


    for item in kept:

        cleaned[
            item["symbol_name"]
        ].append({

            "x": item["x"],
            "y": item["y"],
            "w": item["w"],
            "h": item["h"],
            "score": item["score"]

        })


    return cleaned


# ============================================================
# 進行符號辨識
# ============================================================

def detect_symbols(image):

    if image is None:

        return {}


    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )


    templates = load_symbol_templates()


    detections = {}


    for template_data in templates:

        symbol_name = template_data["name"]

        template = template_data["image"]


        template_height = template.shape[0]

        template_width = template.shape[1]


        if (

            template_height > gray.shape[0]

            or

            template_width > gray.shape[1]

        ):

            continue


        result = cv2.matchTemplate(

            gray,

            template,

            cv2.TM_CCOEFF_NORMED

        )


        # ====================================================
        # 辨識門檻
        # ====================================================

        threshold = 0.62


        locations = np.where(
            result >= threshold
        )


        matches = []


        for y, x in zip(

            locations[0],

            locations[1]

        ):

            score = float(
                result[y, x]
            )


            matches.append({

                "x": int(x),

                "y": int(y),

                "w": int(template_width),

                "h": int(template_height),

                "score": score

            })


        matches = remove_duplicate_matches(
            matches
        )


        detections[
            symbol_name
        ] = matches


    detections = remove_cross_class_duplicates(
        detections
    )


    return detections


# ============================================================
# 計算全部辨識數量
# ============================================================

def count_detections(detections):

    total = 0


    for matches in detections.values():

        total += len(matches)


    return total


# ============================================================
# 圖面畫紅圈
# ============================================================

def draw_detected_symbols(
    image,
    detections
):

    result = image.copy()


    number = 1


    for symbol_name, matches in detections.items():

        for match in matches:

            x = match["x"]
            y = match["y"]

            w = match["w"]
            h = match["h"]


            center_x = x + w // 2
            center_y = y + h // 2


            radius = max(

                10,

                int(
                    max(w, h)
                    *
                    0.60
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

                (
                    label_x,
                    label_y
                ),

                11,

                (
                    0,
                    0,
                    255
                ),

                -1

            )


            label = str(number)


            text_size, _ = cv2.getTextSize(

                label,

                cv2.FONT_HERSHEY_SIMPLEX,

                0.40,

                1

            )


            text_x = int(

                label_x
                -
                text_size[0] / 2

            )

            text_y = int(

                label_y
                +
                text_size[1] / 2

            )


            cv2.putText(

                result,

                label,

                (
                    text_x,
                    text_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.40,

                (
                    255,
                    255,
                    255
                ),

                1,

                cv2.LINE_AA

            )


            number += 1


    return result


# ============================================================
# 首頁
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    return render_template_string(

        PAGE_HTML,

        result=None,

        error=None

    )


# ============================================================
# 分析圖面
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    original_file = request.files.get(
        "original_file"
    )

    modified_file = request.files.get(
        "modified_file"
    )


    # ========================================================
    # 取得使用者選擇的頁數
    # ========================================================

    try:

        original_page = int(
            request.form.get(
                "original_page",
                1
            )
        )

    except ValueError:

        original_page = 1


    try:

        modified_page = int(
            request.form.get(
                "modified_page",
                1
            )
        )

    except ValueError:

        modified_page = 1


    # ========================================================
    # 驗證頁數
    # ========================================================

    if original_page < 1:

        return render_template_string(

            PAGE_HTML,

            result=None,

            error="原始圖頁數必須從 1 開始。"

        )


    if modified_page < 1:

        return render_template_string(

            PAGE_HTML,

            result=None,

            error="修改後圖頁數必須從 1 開始。"

        )


    # ========================================================
    # 驗證檔案
    # ========================================================

    if original_file is None:

        return render_template_string(

            PAGE_HTML,

            result=None,

            error="請上傳原始圖。"

        )


    if modified_file is None:

        return render_template_string(

            PAGE_HTML,

            result=None,

            error="請上傳修改後圖。"

        )


    if not allowed_file(
        original_file.filename
    ):

        return render_template_string(

            PAGE_HTML,

            result=None,

            error="原始圖格式不支援。"

        )


    if not allowed_file(
        modified_file.filename
    ):

        return render_template_string(

            PAGE_HTML,

            result=None,

            error="修改後圖格式不支援。"

        )


    # ========================================================
    # 讀取原始圖指定頁
    # ========================================================

    original_image, original_error = (
        load_image_from_upload(

            original_file,

            original_page

        )
    )


    if original_image is None:

        return render_template_string(

            PAGE_HTML,

            result=None,

            error=(
                "原始圖無法讀取："
                +
                str(original_error)
            )

        )


    # ========================================================
    # 讀取修改圖指定頁
    # ========================================================

    modified_image, modified_error = (
        load_image_from_upload(

            modified_file,

            modified_page

        )
    )


    if modified_image is None:

        return render_template_string(

            PAGE_HTML,

            result=None,

            error=(
                "修改後圖無法讀取："
                +
                str(modified_error)
            )

        )


    # ========================================================
    # 原圖單獨辨識
    # ========================================================

    original_detections = detect_symbols(
        original_image
    )


    # ========================================================
    # 修改圖單獨辨識
    # ========================================================

    modified_detections = detect_symbols(
        modified_image
    )


    # ========================================================
    # 各自計算數量
    # ========================================================

    original_total = count_detections(
        original_detections
    )


    modified_total = count_detections(
        modified_detections
    )


    added = max(

        0,

        modified_total
        -
        original_total

    )


    removed = max(

        0,

        original_total
        -
        modified_total

    )


    net_change = (

        modified_total
        -
        original_total

    )


    # ========================================================
    # 畫辨識結果
    # ========================================================

    original_marked = draw_detected_symbols(

        original_image,

        original_detections

    )


    modified_marked = draw_detected_symbols(

        modified_image,

        modified_detections

    )


    # ========================================================
    # 回傳結果
    # ========================================================

    result = {

        "original_total": original_total,

        "modified_total": modified_total,

        "added": added,

        "removed": removed,

        "net_change": (
            f"{net_change:+d}"
        ),

        "original_image": image_to_base64(
            original_marked
        ),

        "modified_image": image_to_base64(
            modified_marked
        )

    }


    return render_template_string(

        PAGE_HTML,

        result=result,

        error=None

    )


# ============================================================
# Render 啟動
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

        port=port,

        debug=False

    )

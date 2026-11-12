import os
import base64
from pathlib import Path

import cv2
import fitz
import numpy as np

from flask import (
    Flask,
    render_template_string,
    request,
    jsonify
)


# ============================================================
# 基本設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SYMBOL_DIR = BASE_DIR / "static" / "symbols"

ALLOWED_EXTENSIONS = {
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff"
}


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ============================================================
# 13 種圖案名稱
#
# 如果你之後要改中文名稱，
# 只需要改這裡，不影響辨識。
# ============================================================

SYMBOL_NAMES = {

    0: "設備 0",

    1: "設備 1",

    2: "設備 2",

    3: "設備 3",

    4: "設備 4",

    5: "設備 5",

    6: "設備 6",

    7: "設備 7",

    8: "設備 8",

    9: "設備 9",

    10: "設備 10",

    11: "設備 11",

    12: "設備 12"
}


# ============================================================
# HTML
# ============================================================

INDEX_HTML = r"""
<!DOCTYPE html>

<html lang="zh-Hant">

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0"
>

<title>DrawDiff</title>

<style>

*{
    box-sizing:border-box;
}

body{
    margin:0;
    padding:25px;
    font-family:
        Arial,
        "Microsoft JhengHei",
        "Noto Sans TC",
        sans-serif;

    background:#f3f5f7;
    color:#1d2935;
}

.container{
    max-width:1600px;
    margin:auto;
}

h1{
    margin:0 0 8px 0;
}

.subtitle{
    color:#667785;
    margin-bottom:25px;
}

.card{
    background:white;
    border-radius:14px;
    padding:22px;
    margin-bottom:20px;
    box-shadow:
        0 2px 10px
        rgba(0,0,0,0.06);
}

.upload-grid{

    display:grid;

    grid-template-columns:
        1fr
        1fr;

    gap:20px;
}

.upload-box{

    border:
        2px dashed
        #c9d3da;

    border-radius:12px;

    padding:20px;

    background:#fafcfd;
}

.upload-box h3{
    margin-top:0;
}

input[type=file]{

    width:100%;

    margin-top:12px;
}

.page-box{

    margin-top:15px;

    display:flex;

    align-items:center;

    gap:10px;
}

input[type=number]{

    width:100px;

    padding:8px;

    border:
        1px solid
        #c9d3da;

    border-radius:7px;
}

button{

    margin-top:20px;

    padding:
        12px
        24px;

    border:none;

    border-radius:9px;

    font-size:16px;

    cursor:pointer;

    background:#17627f;

    color:white;
}

button:disabled{

    opacity:0.5;

    cursor:not-allowed;
}

#status{

    margin-top:15px;

    font-size:15px;

    color:#566773;
}

.error{

    color:#b42318 !important;
}

#result{

    display:none;
}

.summary{

    display:grid;

    grid-template-columns:
        repeat(4,1fr);

    gap:12px;

    margin-bottom:20px;
}

.summary-box{

    border-radius:10px;

    padding:15px;

    background:#edf3f6;
}

.summary-label{

    font-size:13px;

    color:#687987;
}

.summary-value{

    font-size:30px;

    font-weight:bold;

    margin-top:5px;
}

.images{

    display:grid;

    grid-template-columns:
        1fr
        1fr;

    gap:20px;

    margin-top:20px;
}

.image-box{

    border:
        1px solid
        #d7e0e6;

    border-radius:10px;

    padding:12px;

    background:#fafcfd;
}

.image-title{

    text-align:center;

    font-weight:bold;

    margin-bottom:10px;
}

.image-box img{

    width:100%;

    display:block;

    border:
        1px solid
        #e0e6ea;
}

table{

    width:100%;

    border-collapse:collapse;

    margin-top:25px;
}

th,
td{

    border-bottom:
        1px solid
        #dfe6ea;

    padding:12px;

    text-align:center;
}

th{

    background:#edf3f6;
}

td:first-child,
th:first-child{

    text-align:left;
}

.plus{

    color:#13834d;

    font-weight:bold;
}

.minus{

    color:#c0352b;

    font-weight:bold;
}

.total-row{

    font-weight:bold;

    background:#f1f5f7;
}

@media(max-width:900px){

    .upload-grid,
    .images{

        grid-template-columns:
            1fr;
    }

    .summary{

        grid-template-columns:
            1fr
            1fr;
    }
}

</style>

</head>


<body>

<div class="container">

<h1>DrawDiff｜客變插座辨識</h1>

<div class="subtitle">

使用 static/symbols/0.png ～ 12.png
直接辨識左右兩張圖面，
逐項計算增加與減少。

</div>


<!-- ===================================================== -->
<!-- 上傳 -->
<!-- ===================================================== -->

<div class="card">

<form id="uploadForm">

<div class="upload-grid">


<!-- 原圖 -->

<div class="upload-box">

<h3>原始圖</h3>

<input
type="file"
id="oldFile"
name="old"
accept=".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff"
required
>

<div class="page-box">

<label>PDF 頁碼：</label>

<input
type="number"
id="oldPage"
name="old_page"
value="4"
min="1"
>

</div>

</div>


<!-- 修改圖 -->

<div class="upload-box">

<h3>修改後圖</h3>

<input
type="file"
id="newFile"
name="new"
accept=".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff"
required
>

<div class="page-box">

<label>PDF 頁碼：</label>

<input
type="number"
id="newPage"
name="new_page"
value="3"
min="1"
>

</div>

</div>


</div>


<button
id="analyzeButton"
type="submit"
>

開始辨識

</button>

</form>


<div id="status"></div>

</div>


<!-- ===================================================== -->
<!-- 結果 -->
<!-- ===================================================== -->

<div
class="card"
id="result"
>


<h2>辨識結果</h2>


<div class="summary">


<div class="summary-box">

<div class="summary-label">
原圖總數
</div>

<div
class="summary-value"
id="oldTotal"
>
0
</div>

</div>



<div class="summary-box">

<div class="summary-label">
修改圖總數
</div>

<div
class="summary-value"
id="newTotal"
>
0
</div>

</div>



<div class="summary-box">

<div class="summary-label">
增加
</div>

<div
class="summary-value"
id="addedTotal"
>
0
</div>

</div>



<div class="summary-box">

<div class="summary-label">
減少
</div>

<div
class="summary-value"
id="removedTotal"
>
0
</div>

</div>


</div>



<!-- 左右圖 -->

<div class="images">


<div class="image-box">

<div class="image-title">
原圖辨識結果
</div>

<img
id="oldResult"
alt="原圖"
>

</div>



<div class="image-box">

<div class="image-title">
修改圖辨識結果
</div>

<img
id="newResult"
alt="修改圖"
>

</div>


</div>



<!-- 表格 -->

<table>

<thead>

<tr>

<th>設備類別</th>

<th>原圖</th>

<th>修改圖</th>

<th>增加</th>

<th>減少</th>

<th>淨增減</th>

</tr>

</thead>


<tbody
id="resultRows"
>
</tbody>


</table>


</div>

</div>


<script>


const form =
document.getElementById(
    "uploadForm"
);


const statusEl =
document.getElementById(
    "status"
);


const analyzeButton =
document.getElementById(
    "analyzeButton"
);



form.addEventListener(

    "submit",

    async function(event){

        event.preventDefault();


        const oldFile =
        document.getElementById(
            "oldFile"
        ).files[0];


        const newFile =
        document.getElementById(
            "newFile"
        ).files[0];


        if(!oldFile || !newFile){

            statusEl.textContent =
            "請同時選擇原圖與修改圖";

            statusEl.className =
            "error";

            return;
        }


        const formData =
        new FormData(form);


        analyzeButton.disabled =
        true;


        statusEl.textContent =
        "正在辨識 13 種設備圖案...";


        statusEl.className =
        "";


        try{


            const response =
            await fetch(

                "/analyze",

                {
                    method:"POST",

                    body:formData
                }
            );


            const data =
            await response.json();


            if(!response.ok){

                throw new Error(

                    data.error
                    ||
                    "辨識失敗"
                );
            }


            document.getElementById(
                "result"
            ).style.display =
            "block";


            document.getElementById(
                "oldTotal"
            ).textContent =
            data.old_count;


            document.getElementById(
                "newTotal"
            ).textContent =
            data.new_count;


            document.getElementById(
                "addedTotal"
            ).textContent =
            data.added_count;


            document.getElementById(
                "removedTotal"
            ).textContent =
            data.removed_count;


            document.getElementById(
                "oldResult"
            ).src =
            data.old_image;


            document.getElementById(
                "newResult"
            ).src =
            data.new_image;



            const rows =
            document.getElementById(
                "resultRows"
            );


            rows.innerHTML =
            "";


            data.rows.forEach(

                function(row){


                    const tr =
                    document.createElement(
                        "tr"
                    );


                    let diffText =
                    row.difference;


                    if(
                        row.difference > 0
                    ){

                        diffText =
                        "+" +
                        row.difference;
                    }


                    tr.innerHTML = `

<td>
${row.name}
</td>

<td>
${row.old_count}
</td>

<td>
${row.new_count}
</td>

<td class="plus">
${row.added_count}
</td>

<td class="minus">
${row.removed_count}
</td>

<td>
${diffText}
</td>

`;


                    rows.appendChild(
                        tr
                    );


                }

            );


            // ================================
            // 合計
            // ================================


            const totalRow =
            document.createElement(
                "tr"
            );


            totalRow.className =
            "total-row";


            totalRow.innerHTML = `

<td>
合計
</td>

<td>
${data.old_count}
</td>

<td>
${data.new_count}
</td>

<td>
${data.added_count}
</td>

<td>
${data.removed_count}
</td>

<td>
${data.difference > 0 ? "+" : ""}
${data.difference}
</td>

`;


            rows.appendChild(
                totalRow
            );


            statusEl.textContent =
            "辨識完成";


            document.getElementById(
                "result"
            ).scrollIntoView({

                behavior:"smooth"
            });


        }
        catch(error){


            statusEl.textContent =
            error.message;


            statusEl.className =
            "error";


        }
        finally{


            analyzeButton.disabled =
            false;


        }


    }

);


</script>

</body>

</html>
"""


# ============================================================
# 副檔名
# ============================================================

def allowed(filename):

    if "." not in filename:

        return False

    ext = (
        filename
        .rsplit(".", 1)[1]
        .lower()
    )

    return ext in ALLOWED_EXTENSIONS



# ============================================================
# 讀取圖片 / PDF 指定頁
# ============================================================

def read_upload(
    file,
    page_number=1
):

    filename =
    file.filename or ""


    if "." not in filename:

        return None


    ext = (
        filename
        .rsplit(".", 1)[1]
        .lower()
    )


    raw =
    file.read()


    if not raw:

        return None


    # ========================================================
    # PDF
    # ========================================================

    if ext == "pdf":


        doc = fitz.open(

            stream=raw,

            filetype="pdf"

        )


        try:


            page_index =
            page_number - 1


            if page_index < 0:

                page_index = 0


            if page_index >= len(doc):

                raise ValueError(

                    f"PDF 只有 "
                    f"{len(doc)} 頁，"
                    f"找不到第 "
                    f"{page_number} 頁"

                )


            page =
            doc.load_page(
                page_index
            )


            # 4 倍解析度
            # 小型插座符號比較容易辨識

            pix =
            page.get_pixmap(

                matrix=fitz.Matrix(
                    4.0,
                    4.0
                ),

                alpha=False

            )


            png_bytes =
            pix.tobytes(
                "png"
            )


        finally:


            doc.close()


        arr =
        np.frombuffer(

            png_bytes,

            np.uint8

        )


        image =
        cv2.imdecode(

            arr,

            cv2.IMREAD_COLOR

        )


        return image


    # ========================================================
    # 一般圖片
    # ========================================================


    arr =
    np.frombuffer(

        raw,

        np.uint8

    )


    image =
    cv2.imdecode(

        arr,

        cv2.IMREAD_COLOR

    )


    return image



# ============================================================
# 去除白邊
# ============================================================

def crop_white_border(image):

    gray =
    cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )


    mask =
    gray < 245


    coords =
    cv2.findNonZero(

        mask.astype(
            np.uint8
        )
    )


    if coords is None:

        return image


    x, y, w, h =
    cv2.boundingRect(
        coords
    )


    # 保留一點邊界

    padding = 10


    x =
    max(
        0,
        x - padding
    )


    y =
    max(
        0,
        y - padding
    )


    x2 =
    min(
        image.shape[1],
        x + w + padding * 2
    )


    y2 =
    min(
        image.shape[0],
        y + h + padding * 2
    )


    return image[
        y:y2,
        x:x2
    ]



# ============================================================
# 圖面預處理
# ============================================================

def prepare_plan(image):

    gray =
    cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )


    gray =
    cv2.GaussianBlur(

        gray,

        (3, 3),

        0

    )


    return gray



# ============================================================
# 讀取 13 個樣板
# ============================================================

def load_symbol_templates():

    templates = {}

    missing = []


    for symbol_id in range(13):


        path =
        SYMBOL_DIR / (
            f"{symbol_id}.png"
        )


        if not path.exists():

            missing.append(
                str(path)
            )

            continue


        image =
        cv2.imread(

            str(path),

            cv2.IMREAD_UNCHANGED

        )


        if image is None:

            missing.append(
                str(path)
            )

            continue


        # PNG 有透明背景

        if len(image.shape) == 3 and image.shape[2] == 4:


            alpha =
            image[:, :, 3]


            bgr =
            image[:, :, :3]


            gray =
            cv2.cvtColor(

                bgr,

                cv2.COLOR_BGR2GRAY

            )


            # 透明區域變白

            gray[
                alpha == 0
            ] = 255


        else:


            if len(image.shape) == 3:


                gray =
                cv2.cvtColor(

                    image,

                    cv2.COLOR_BGR2GRAY

                )


            else:


                gray =
                image


        # 二值化

        gray =
        cv2.GaussianBlur(

            gray,

            (3, 3),

            0

        )


        templates[symbol_id] =
        gray


    return templates, missing



# ============================================================
# 旋轉樣板
# ============================================================

def rotate_image(
    image,
    angle
):


    if angle == 0:

        return image


    if angle == 90:

        return cv2.rotate(

            image,

            cv2.ROTATE_90_CLOCKWISE

        )


    if angle == 180:

        return cv2.rotate(

            image,

            cv2.ROTATE_180

        )


    return cv2.rotate(

        image,

        cv2.ROTATE_90_COUNTERCLOCKWISE

    )



# ============================================================
# 計算 IoU
# ============================================================

def calculate_iou(
    a,
    b
):

    ax1 =
    a["x"]

    ay1 =
    a["y"]

    ax2 =
    ax1 + a["w"]

    ay2 =
    ay1 + a["h"]


    bx1 =
    b["x"]

    by1 =
    b["y"]

    bx2 =
    bx1 + b["w"]

    by2 =
    by1 + b["h"]


    x1 =
    max(ax1, bx1)

    y1 =
    max(ay1, by1)

    x2 =
    min(ax2, bx2)

    y2 =
    min(ay2, by2)


    inter_w =
    max(
        0,
        x2 - x1
    )


    inter_h =
    max(
        0,
        y2 - y1
    )


    intersection =
    inter_w * inter_h


    area_a =
    a["w"] * a["h"]


    area_b =
    b["w"] * b["h"]


    union =
    area_a + area_b - intersection


    if union <= 0:

        return 0


    return intersection / union



# ============================================================
# 同一種類去重
# ============================================================

def remove_duplicates(matches):


    matches =
    sorted(

        matches,

        key=lambda item:
        item["score"],

        reverse=True

    )


    kept = []


    for candidate in matches:


        duplicate = False


        for existing in kept:


            overlap =
            calculate_iou(

                candidate,

                existing

            )


            if overlap >= 0.30:


                duplicate = True

                break


        if not duplicate:

            kept.append(
                candidate
            )


    return kept



# ============================================================
# 不同類別重複辨識
#
# 保留相似度最高的結果
# ============================================================

def remove_cross_class_duplicates(
    detections
):


    all_items = []


    for symbol_id, matches in detections.items():


        for item in matches:


            current =
            dict(item)


            current["symbol_id"] =
            symbol_id


            all_items.append(
                current
            )


    # 相似度高的優先保留

    all_items.sort(

        key=lambda item:
        item["score"],

        reverse=True

    )


    kept = []


    for candidate in all_items:


        duplicate = False


        cx = (

            candidate["x"]

            + candidate["w"] / 2

        )


        cy = (

            candidate["y"]

            + candidate["h"] / 2

        )


        for existing in kept:


            ex = (

                existing["x"]

                + existing["w"] / 2

            )


            ey = (

                existing["y"]

                + existing["h"] / 2

            )


            # 中心距離

            distance = (

                (cx - ex) ** 2

                +

                (cy - ey) ** 2

            ) ** 0.5


            candidate_size = max(

                candidate["w"],

                candidate["h"]

            )


            existing_size = max(

                existing["w"],

                existing["h"]

            )


            min_size = min(

                candidate_size,

                existing_size

            )


            overlap =
            calculate_iou(

                candidate,

                existing

            )


            # 必須同時：
            #
            # 1. 框有一定重疊
            # 2. 中心位置非常接近

            if (

                overlap >= 0.35

                and

                distance <=
                max(
                    3,
                    min_size * 0.25
                )

            ):


                duplicate = True

                break


        if not duplicate:

            kept.append(
                candidate
            )


    cleaned = {

        symbol_id: []

        for symbol_id in detections

    }


    for item in kept:


        cleaned[
            item["symbol_id"]
        ].append({

            "x":
            item["x"],

            "y":
            item["y"],

            "w":
            item["w"],

            "h":
            item["h"],

            "score":
            item["score"]

        })


    return cleaned



# ============================================================
# 偵測 13 種圖案
# ============================================================

def detect_symbols(
    image,
    templates
):


    gray =
    prepare_plan(
        image
    )


    original_h, original_w =
    gray.shape


    # ========================================================
    # 大圖縮小
    # ========================================================

    ratio =
    min(

        1.0,

        1600 /
        max(
            original_h,
            original_w
        )

    )


    if ratio < 1.0:


        search_w =
        max(

            1,

            int(
                original_w * ratio
            )

        )


        search_h =
        max(

            1,

            int(
                original_h * ratio
            )

        )


        gray =
        cv2.resize(

            gray,

            (
                search_w,
                search_h
            ),

            interpolation=
            cv2.INTER_AREA

        )


    scale_x = (
        original_w /
        gray.shape[1]
    )


    scale_y = (
        original_h /
        gray.shape[0]
    )


    # ========================================================
    # 各品項門檻
    # ========================================================

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


    # ========================================================
    # 最小縮放
    # ========================================================

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


    # ========================================================
    # 縮放比例
    #
    # 保留較多尺寸，
    # 避免圖面大小不同找不到符號
    # ========================================================

    scale_values = [

        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.15,
        1.30,
        1.50,
        1.70,
        2.00

    ]


    # ========================================================
    # 旋轉角度
    # ========================================================

    angles = [

        0,
        90,
        180,
        270

    ]


    detections = {

        symbol_id: []

        for symbol_id in templates

    }


    # ========================================================
    # 每一個符號
    # ========================================================

    for symbol_id, template in templates.items():


        threshold =
        thresholds.get(
            symbol_id,
            0.60
        )


        min_scale =
        min_scales.get(
            symbol_id,
            0.40
        )


        # ====================================================
        # 多尺寸
        # ====================================================

        for scale in scale_values:


            if scale < min_scale:

                continue


            scaled_w =
            max(
                1,
                int(
                    template.shape[1]
                    * scale
                )
            )


            scaled_h =
            max(
                1,
                int(
                    template.shape[0]
                    * scale
                )
            )


            if (

                scaled_w >= gray.shape[1]

                or

                scaled_h >= gray.shape[0]

            ):

                continue


            scaled =
            cv2.resize(

                template,

                (
                    scaled_w,
                    scaled_h
                ),

                interpolation=
                cv2.INTER_CUBIC

            )


            # =================================================
            # 多方向
            # =================================================

            for angle in angles:


                rotated =
                rotate_image(

                    scaled,

                    angle

                )


                rh, rw =
                rotated.shape


                if (

                    rw >= gray.shape[1]

                    or

                    rh >= gray.shape[0]

                ):

                    continue


                result =
                cv2.matchTemplate(

                    gray,

                    rotated,

                    cv2.TM_CCOEFF_NORMED

                )


                locations =
                np.where(

                    result >= threshold

                )


                for y, x in zip(

                    locations[0],

                    locations[1]

                ):


                    score =
                    float(
                        result[y, x]
                    )


                    detections[
                        symbol_id
                    ].append({

                        "x":
                        int(
                            x * scale_x
                        ),

                        "y":
                        int(
                            y * scale_y
                        ),

                        "w":
                        int(
                            rw * scale_x
                        ),

                        "h":
                        int(
                            rh * scale_y
                        ),

                        "score":
                        score

                    })


        # ====================================================
        # 同類別去重
        # ====================================================

        detections[
            symbol_id
        ] = remove_duplicates(

            detections[
                symbol_id
            ]

        )


    return detections



# ============================================================
# 在圖面上畫紅圈與類別編號
# ============================================================

def draw_detected_symbols(
    image,
    detections
):


    result =
    image.copy()


    for symbol_id, matches in detections.items():


        for match in matches:


            x =
            match["x"]


            y =
            match["y"]


            w =
            match["w"]


            h =
            match["h"]


            center_x =
            x + w // 2


            center_y =
            y + h // 2


            radius =
            max(

                10,

                int(
                    max(w, h)
                    * 0.60
                )

            )


            # ================================================
            # 紅圈
            # ================================================

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


            # ================================================
            # 編號
            # ================================================

            label =
            str(
                symbol_id
            )


            label_x =
            max(

                12,

                center_x - radius

            )


            label_y =
            max(

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


            text_size, _ =
            cv2.getTextSize(

                label,

                cv2.FONT_HERSHEY_SIMPLEX,

                0.42,

                1

            )


            text_x =
            int(

                label_x
                -
                text_size[0] / 2

            )


            text_y =
            int(

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

                0.42,

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
# 圖片轉 Base64
# ============================================================

def image_to_base64(image):


    h, w =
    image.shape[:2]


    max_side = 2200


    ratio =
    min(

        1.0,

        max_side /
        max(
            h,
            w
        )

    )


    if ratio < 1.0:


        image =
        cv2.resize(

            image,

            (
                int(w * ratio),
                int(h * ratio)
            ),

            interpolation=
            cv2.INTER_AREA

        )


    success, buffer =
    cv2.imencode(

        ".jpg",

        image,

        [

            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),

            90

        ]

    )


    if not success:

        raise RuntimeError(
            "圖片編碼失敗"
        )


    encoded =
    base64.b64encode(

        buffer.tobytes()

    ).decode(
        "utf-8"
    )


    return (
        "data:image/jpeg;base64,"
        +
        encoded
    )



# ============================================================
# 首頁
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        INDEX_HTML
    )



# ============================================================
# 分析 API
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze_api():


    old_file =
    request.files.get(
        "old"
    )


    new_file =
    request.files.get(
        "new"
    )


    if not old_file or not new_file:


        return jsonify(

            error=
            "請同時上傳原始圖與修改後圖"

        ), 400



    if (

        not allowed(
            old_file.filename
        )

        or

        not allowed(
            new_file.filename
        )

    ):


        return jsonify(

            error=
            "不支援此檔案格式"

        ), 400



    # ========================================================
    # 頁碼
    # ========================================================

    try:


        old_page =
        int(

            request.form.get(

                "old_page",

                1

            )

        )


        new_page =
        int(

            request.form.get(

                "new_page",

                1

            )

        )


        if old_page < 1:

            old_page = 1


        if new_page < 1:

            new_page = 1


    except Exception:


        return jsonify(

            error=
            "PDF 頁碼格式錯誤"

        ), 400



    try:


        # ====================================================
        # 讀取圖面
        # ====================================================

        old_image =
        read_upload(

            old_file,

            old_page

        )


        new_image =
        read_upload(

            new_file,

            new_page

        )


        if (

            old_image is None

            or

            new_image is None

        ):


            raise ValueError(
                "圖片無法讀取"
            )



        # ====================================================
        # 裁白邊
        # ====================================================

        old_image =
        crop_white_border(
            old_image
        )


        new_image =
        crop_white_border(
            new_image
        )



        # ====================================================
        # 載入 13 種樣板
        # ====================================================

        templates, missing =
        load_symbol_templates()


        if missing:


            raise ValueError(

                "找不到圖案樣板："

                +

                "、".join(
                    missing
                )

            )



        # ====================================================
        # 原圖辨識
        # ====================================================

        old_detections =
        detect_symbols(

            old_image,

            templates

        )



        # ====================================================
        # 修改圖辨識
        # ====================================================

        new_detections =
        detect_symbols(

            new_image,

            templates

        )



        # ====================================================
        # 跨類別去重
        # ====================================================

        old_detections =
        remove_cross_class_duplicates(

            old_detections

        )


        new_detections =
        remove_cross_class_duplicates(

            new_detections

        )



        # ====================================================
        # 統計 13 種
        # ====================================================

        rows = []


        old_total = 0

        new_total = 0

        added_total = 0

        removed_total = 0



        for symbol_id in range(13):


            old_count =
            len(

                old_detections.get(

                    symbol_id,

                    []

                )

            )


            new_count =
            len(

                new_detections.get(

                    symbol_id,

                    []

                )

            )


            difference =
            new_count - old_count


            added_count =
            max(

                difference,

                0

            )


            removed_count =
            max(

                -difference,

                0

            )



            old_total += (
                old_count
            )


            new_total += (
                new_count
            )


            added_total += (
                added_count
            )


            removed_total += (
                removed_count
            )



            rows.append({

                "id":
                symbol_id,

                "name":
                SYMBOL_NAMES.get(

                    symbol_id,

                    f"設備 {symbol_id}"

                ),

                "old_count":
                old_count,

                "new_count":
                new_count,

                "added_count":
                added_count,

                "removed_count":
                removed_count,

                "difference":
                difference

            })



        # ====================================================
        # 畫辨識結果
        # ====================================================

        old_marked =
        draw_detected_symbols(

            old_image,

            old_detections

        )


        new_marked =
        draw_detected_symbols(

            new_image,

            new_detections

        )



        # ====================================================
        # 回傳
        # ====================================================

        return jsonify(


            old_count=
            old_total,


            new_count=
            new_total,


            difference=
            new_total
            -
            old_total,


            added_count=
            added_total,


            removed_count=
            removed_total,


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



    except Exception as error:


        app.logger.exception(
            "分析失敗"
        )


        return jsonify(

            error=
            str(error)

        ), 500



# ============================================================
# 啟動
# ============================================================

if __name__ == "__main__":


    port =
    int(

        os.environ.get(

            "PORT",

            10000

        )

    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=True

    )

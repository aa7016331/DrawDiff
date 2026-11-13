import os
import csv
import io
import base64
from pathlib import Path

import cv2
import fitz
import numpy as np

from flask import (
    Flask,
    render_template_string,
    request,
    jsonify,
    Response
)


# ============================================================
# 基本設定
# ============================================================

BASE = Path(__file__).resolve().parent

SYMBOL_DIR = BASE / "static" / "symbols"

ALLOWED = {
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff"
}

RENDER_SCALE = 3.0

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024


# ============================================================
# HTML
# ============================================================

INDEX_HTML = r'''
<!DOCTYPE html>
<html lang="zh-Hant">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>DrawDiff｜設備分類增減辨識</title>

<style>

:root{
    font-family:
        Arial,
        "Microsoft JhengHei",
        "Noto Sans TC",
        sans-serif;

    color:#152432;
    background:#f2f5f7;
}

*{
    box-sizing:border-box;
}

body{
    margin:0;
    padding:24px;
}

main{
    max-width:1500px;
    margin:auto;
}

header{
    max-width:1500px;
    margin:0 auto 20px;
}

h1{
    margin:0 0 6px;
    font-size:26px;
}

.sub{
    color:#627381;
    font-size:14px;
}

.card{
    background:#fff;

    border:1px solid #d7e0e6;

    border-radius:14px;

    padding:22px;

    margin-bottom:18px;
}

h2{
    margin:0 0 18px;
}

.upload-grid{

    display:grid;

    grid-template-columns:
        1fr 1fr;

    gap:16px;
}

.upload-box{

    border:1px dashed #aabac5;

    background:#fbfcfd;

    padding:16px;

    border-radius:10px;
}

.upload-box strong{

    display:block;

    margin-bottom:8px;
}

input[type=file]{

    width:100%;

    margin-top:8px;
}

.page-row{

    display:flex;

    align-items:center;

    gap:10px;

    margin-top:12px;

    color:#546675;

    font-size:14px;
}

input[type=number]{

    width:80px;

    padding:8px;

    border:1px solid #cbd6dd;

    border-radius:8px;
}

.actions{

    display:flex;

    gap:12px;

    margin-top:18px;

    flex-wrap:wrap;
}

button{

    border:0;

    border-radius:9px;

    padding:11px 18px;

    font-size:15px;

    cursor:pointer;
}

.primary{

    background:#164f71;

    color:white;
}

.secondary{

    background:#e9f0f4;

    color:#1b3a4e;
}

button:disabled{

    opacity:.5;

    cursor:not-allowed;
}

#status{

    margin-top:14px;

    min-height:20px;

    color:#5d6c77;

    font-size:14px;
}

.error{

    color:#b42318 !important;
}

#result{

    display:none;
}

.summary{

    display:flex;

    gap:12px;

    flex-wrap:wrap;

    margin-bottom:18px;
}

.pill{

    background:#edf3f6;

    border-radius:999px;

    padding:10px 15px;

    font-weight:600;
}

.pill.plus{

    color:#13724e;

    background:#e8f5ee;
}

.pill.minus{

    color:#a33c32;

    background:#faeeec;
}

.compare-grid{

    display:grid;

    grid-template-columns:
        1fr 1fr;

    gap:16px;
}

.image-panel{

    border:1px solid #d7e0e6;

    background:#fafcfd;

    border-radius:10px;

    padding:12px;

    min-width:0;
}

.image-title{

    text-align:center;

    font-weight:700;

    margin-bottom:8px;
}

.image-panel img{

    width:100%;

    height:auto;

    display:block;

    border:1px solid #e5eaee;
}

table{

    width:100%;

    border-collapse:collapse;

    margin-top:22px;
}

th,
td{

    border-bottom:
        1px solid #e3e9ed;

    padding:12px;

    text-align:center;
}

th{

    background:#edf2f5;
}

td:first-child,
th:first-child{

    text-align:left;
}

.plus-text{

    color:#13724e;

    font-weight:700;
}

.minus-text{

    color:#a33c32;

    font-weight:700;
}

.total-row{

    font-weight:700;

    background:#f5f8fa;
}

.note{

    margin-top:14px;

    color:#687987;

    font-size:13px;

    line-height:1.5;
}

@media(max-width:850px){

    body{
        padding:12px;
    }

    .upload-grid,
    .compare-grid{

        grid-template-columns:
            1fr;
    }

}

</style>

</head>


<body>

<header>

<h1>
DrawDiff｜設備分類增減辨識
</h1>

<div class="sub">

依照圖面符號分類，
逐項統計原圖與修改圖的設備數量。

</div>

</header>


<main>


<section class="card">

<h2>上傳圖面</h2>


<form id="uploadForm">


<div class="upload-grid">


<div class="upload-box">

<strong>
原始圖
</strong>

<div class="sub">

PDF 請選擇頁碼

</div>


<input
    id="oldFile"
    name="old"
    type="file"
    accept=".pdf,.jpg,.jpeg,.png,.bmp,.tif,.tiff"
    required
>


<div class="page-row">

PDF 頁碼

<input
    id="oldPage"
    name="old_page"
    type="number"
    min="1"
    value="4"
>

</div>

</div>



<div class="upload-box">

<strong>
設計師修改圖
</strong>

<div class="sub">

PDF 請選擇頁碼

</div>


<input
    id="newFile"
    name="new"
    type="file"
    accept=".pdf,.jpg,.jpeg,.png,.bmp,.tif,.tiff"
    required
>


<div class="page-row">

PDF 頁碼

<input
    id="newPage"
    name="new_page"
    type="number"
    min="1"
    value="3"
>

</div>

</div>


</div>


<div class="actions">


<button
    class="primary"
    id="analyzeButton"
    type="submit"
>

開始辨識

</button>


<button
    class="secondary"
    id="exportButton"
    type="button"
    disabled
>

匯出 CSV

</button>


</div>


</form>


<div id="status"></div>


</section>



<section
    class="card"
    id="result"
>


<h2>
設備逐項增減結果
</h2>


<div
    class="summary"
    id="summary"
></div>



<div class="compare-grid">


<div class="image-panel">


<div class="image-title">

原始圖

</div>


<img
    id="oldResult"
>


</div>



<div class="image-panel">


<div class="image-title">

修改後圖

</div>


<img
    id="newResult"
>


</div>


</div>



<table>


<thead>

<tr>

<th>設備項目</th>

<th>原圖</th>

<th>修改後</th>

<th>增加</th>

<th>減少</th>

<th>淨增減</th>

</tr>

</thead>


<tbody
    id="resultRows"
></tbody>


</table>


<div
    class="note"
    id="resultNote"
></div>


</section>


</main>



<script>


let latestRows = [];


const form =
    document.getElementById(
        'uploadForm'
    );


const statusEl =
    document.getElementById(
        'status'
    );


const analyzeButton =
    document.getElementById(
        'analyzeButton'
    );


const exportButton =
    document.getElementById(
        'exportButton'
    );



form.addEventListener(

    'submit',

    async(event)=>{


        event.preventDefault();


        const oldFile =
            document
            .getElementById('oldFile')
            .files[0];


        const newFile =
            document
            .getElementById('newFile')
            .files[0];


        if(
            !oldFile
            ||
            !newFile
        ){

            statusEl.textContent =
                '請同時選擇原始圖及修改後圖。';

            statusEl.className =
                'error';

            return;
        }


        const payload =
            new FormData(form);


        analyzeButton.disabled =
            true;


        exportButton.disabled =
            true;


        statusEl.textContent =
            '正在辨識各設備符號，請稍候...';


        statusEl.className =
            '';


        try{


            const response =
                await fetch(

                    '/analyze',

                    {
                        method:'POST',
                        body:payload
                    }

                );


            const data =
                await response.json();


            if(!response.ok){

                throw new Error(

                    data.error
                    ||
                    '辨識失敗'

                );
            }


            document
            .getElementById('result')
            .style.display =
                'block';


            document
            .getElementById('oldResult')
            .src =
                data.old_image;


            document
            .getElementById('newResult')
            .src =
                data.new_image;


            const summary =
                document
                .getElementById('summary');


            summary.innerHTML =
                '';


            const summaryData = [

                [
                    '原圖總數 '
                    +
                    data.old_total
                    +
                    ' 個',
                    ''
                ],

                [
                    '修改後總數 '
                    +
                    data.new_total
                    +
                    ' 個',
                    ''
                ],

                [
                    '增加 '
                    +
                    data.total_added
                    +
                    ' 個',
                    'plus'
                ],

                [
                    '減少 '
                    +
                    data.total_removed
                    +
                    ' 個',
                    'minus'
                ]

            ];


            summaryData.forEach(

                ([label,cls])=>{

                    const item =
                        document.createElement(
                            'div'
                        );


                    item.className =
                        'pill '
                        +
                        cls;


                    item.textContent =
                        label;


                    summary.appendChild(
                        item
                    );

                }

            );


            const body =
                document
                .getElementById(
                    'resultRows'
                );


            body.innerHTML =
                '';


            data.rows.forEach(

                row=>{


                    const tr =
                        document.createElement(
                            'tr'
                        );


                    const diff =
                        row.difference;


                    const values = [

                        row.name,

                        row.old_count,

                        row.new_count,

                        row.added,

                        row.removed,

                        (
                            diff > 0
                            ?
                            '+'
                            +
                            diff
                            :
                            diff
                        )

                    ];


                    values.forEach(

                        (value,index)=>{


                            const td =
                                document.createElement(
                                    'td'
                                );


                            td.textContent =
                                value;


                            if(
                                index === 5
                                &&
                                diff > 0
                            ){

                                td.className =
                                    'plus-text';

                            }


                            if(
                                index === 5
                                &&
                                diff < 0
                            ){

                                td.className =
                                    'minus-text';

                            }


                            tr.appendChild(
                                td
                            );

                        }

                    );


                    body.appendChild(
                        tr
                    );

                }

            );


            const totalRow =
                document.createElement(
                    'tr'
                );


            totalRow.className =
                'total-row';


            const totalDiff =
                data.new_total
                -
                data.old_total;


            const totalValues = [

                '合計',

                data.old_total,

                data.new_total,

                data.total_added,

                data.total_removed,

                (
                    totalDiff > 0
                    ?
                    '+'
                    +
                    totalDiff
                    :
                    totalDiff
                )

            ];


            totalValues.forEach(

                value=>{

                    const td =
                        document.createElement(
                            'td'
                        );


                    td.textContent =
                        value;


                    totalRow.appendChild(
                        td
                    );

                }

            );


            body.appendChild(
                totalRow
            );


            latestRows =
                data.rows;


            document
            .getElementById(
                'resultNote'
            )
            .textContent =
                '紅圈表示辨識到的設備符號。設備分類依照 static/symbols 資料夾中的模板檔名進行。';


            exportButton.disabled =
                false;


            statusEl.textContent =
                '辨識完成，請核對紅圈位置與分類結果。';


            statusEl.className =
                '';


        }
        catch(err){


            statusEl.textContent =
                err.message;


            statusEl.className =
                'error';


        }
        finally{


            analyzeButton.disabled =
                false;

        }


    }

);



exportButton.addEventListener(

    'click',

    async()=>{


        if(
            !latestRows.length
        ){
            return;
        }


        try{


            const response =
                await fetch(

                    '/export',

                    {

                        method:'POST',

                        headers:{
                            'Content-Type':
                                'application/json'
                        },

                        body:JSON.stringify({

                            rows:latestRows

                        })

                    }

                );


            if(!response.ok){

                throw new Error(
                    'CSV 匯出失敗'
                );

            }


            const blob =
                await response.blob();


            const url =
                URL.createObjectURL(
                    blob
                );


            const a =
                document.createElement(
                    'a'
                );


            a.href =
                url;


            a.download =
                'DrawDiff設備增減統計.csv';


            document.body.appendChild(
                a
            );


            a.click();


            a.remove();


            URL.revokeObjectURL(
                url
            );


        }
        catch(err){


            statusEl.textContent =
                err.message;


            statusEl.className =
                'error';

        }


    }

);


</script>


</body>

</html>
'''


# ============================================================
# 基本工具
# ============================================================

def filename_ext(filename):

    return (
        Path(
            filename
            or ""
        )
        .suffix
        .lower()
        .lstrip(".")
    )


# ============================================================
# 圖片轉 Base64
# ============================================================

def image_to_data_url(
    image,
    max_side=2200,
    quality=85
):

    h, w = image.shape[:2]


    scale = min(
        1.0,
        max_side / max(h, w)
    )


    if scale < 1.0:

        image = cv2.resize(

            image,

            (
                int(w * scale),
                int(h * scale)
            ),

            interpolation=cv2.INTER_AREA

        )


    success, buffer = cv2.imencode(

        ".jpg",

        image,

        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            quality
        ]

    )


    if not success:

        raise RuntimeError(
            "圖片編碼失敗"
        )


    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("ascii")


    return (
        "data:image/jpeg;base64,"
        +
        encoded
    )


# ============================================================
# 讀取 PDF
# ============================================================

def load_pdf_page(
    raw,
    page_number
):

    document = None


    try:


        document = fitz.open(

            stream=raw,

            filetype="pdf"

        )


        page_index =
            page_number - 1


        if(
            page_index < 0
            or
            page_index >= len(document)
        ):

            raise ValueError(

                f"PDF 只有 {len(document)} 頁，"
                f"沒有第 {page_number} 頁。"

            )


        page =
            document.load_page(
                page_index
            )


        pix =
            page.get_pixmap(

                matrix=fitz.Matrix(

                    RENDER_SCALE,

                    RENDER_SCALE

                ),

                alpha=False,

                colorspace=fitz.csRGB

            )


        rgb = np.frombuffer(

            pix.samples,

            dtype=np.uint8

        ).reshape(

            pix.height,

            pix.width,

            pix.n

        )


        image = cv2.cvtColor(

            rgb,

            cv2.COLOR_RGB2BGR

        )


        return image


    finally:


        if document is not None:

            document.close()


# ============================================================
# 讀取上傳檔案
# ============================================================

def load_upload(
    upload,
    page_number
):

    ext =
        filename_ext(
            upload.filename
        )


    if ext not in ALLOWED:

        raise ValueError(
            "不支援的檔案格式。"
        )


    raw =
        upload.read()


    if not raw:

        raise ValueError(
            "上傳檔案是空的。"
        )


    if ext == "pdf":

        return load_pdf_page(

            raw,

            page_number

        )


    array = np.frombuffer(

        raw,

        dtype=np.uint8

    )


    image = cv2.imdecode(

        array,

        cv2.IMREAD_COLOR

    )


    if image is None:

        raise ValueError(
            "圖片無法讀取。"
        )


    return image


# ============================================================
# 讀取所有設備模板
# ============================================================

def load_symbol_templates():

    if not SYMBOL_DIR.exists():

        raise ValueError(

            "找不到 static/symbols 資料夾。"

        )


    templates = []


    allowed_images = {

        ".png",

        ".jpg",

        ".jpeg",

        ".bmp"

    }


    for file_path in sorted(
        SYMBOL_DIR.iterdir()
    ):


        if not file_path.is_file():

            continue


        if(
            file_path.suffix.lower()
            not in allowed_images
        ):

            continue


        image = cv2.imread(

            str(file_path),

            cv2.IMREAD_GRAYSCALE

        )


        if image is None:

            continue


        if(
            image.shape[0] < 5
            or
            image.shape[1] < 5
        ):

            continue


        name =
            file_path.stem


        # 底線顯示成空白
        name =
            name.replace(
                "_",
                " "
            )


        templates.append({

            "name": name,

            "image": image

        })


    if not templates:

        raise ValueError(

            "static/symbols 資料夾內沒有可用的設備模板圖片。"

        )


    return templates


# ============================================================
# IoU
# ============================================================

def calculate_iou(
    a,
    b
):

    ax1 = a["x"]
    ay1 = a["y"]

    ax2 = ax1 + a["w"]
    ay2 = ay1 + a["h"]


    bx1 = b["x"]
    by1 = b["y"]

    bx2 = bx1 + b["w"]
    by2 = by1 + b["h"]


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )


    intersection =
        iw * ih


    area_a =
        (ax2 - ax1)
        *
        (ay2 - ay1)


    area_b =
        (bx2 - bx1)
        *
        (by2 - by1)


    union =
        area_a
        +
        area_b
        -
        intersection


    if union <= 0:

        return 0.0


    return (
        intersection
        /
        union
    )


# ============================================================
# 同類型去重
# ============================================================

def remove_duplicates(
    matches
):

    matches = sorted(

        matches,

        key=lambda x: x["score"],

        reverse=True

    )


    kept = []


    for candidate in matches:


        duplicate = False


        cx = (
            candidate["x"]
            +
            candidate["w"] / 2
        )

        cy = (
            candidate["y"]
            +
            candidate["h"] / 2
        )


        for existing in kept:


            ex = (
                existing["x"]
                +
                existing["w"] / 2
            )

            ey = (
                existing["y"]
                +
                existing["h"] / 2
            )


            distance = (

                (cx - ex) ** 2
                +
                (cy - ey) ** 2

            ) ** 0.5


            overlap =
                calculate_iou(

                    candidate,

                    existing

                )


            size = min(

                max(
                    candidate["w"],
                    candidate["h"]
                ),

                max(
                    existing["w"],
                    existing["h"]
                )

            )


            if(

                overlap >= 0.30

                or

                distance <= max(
                    6,
                    size * 0.35
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
# 不同設備分類去重
# ============================================================

def remove_cross_class_duplicates(
    detections
):

    all_items = []


    for name, matches in detections.items():

        for item in matches:

            data = dict(item)

            data["name"] = name

            all_items.append(
                data
            )


    all_items.sort(

        key=lambda x: x["score"],

        reverse=True

    )


    kept = []


    for candidate in all_items:


        duplicate = False


        cx = (
            candidate["x"]
            +
            candidate["w"] / 2
        )

        cy = (
            candidate["y"]
            +
            candidate["h"] / 2
        )


        for existing in kept:


            ex = (
                existing["x"]
                +
                existing["w"] / 2
            )

            ey = (
                existing["y"]
                +
                existing["h"] / 2
            )


            distance = (

                (cx - ex) ** 2
                +
                (cy - ey) ** 2

            ) ** 0.5


            overlap =
                calculate_iou(

                    candidate,

                    existing

                )


            size = min(

                max(
                    candidate["w"],
                    candidate["h"]
                ),

                max(
                    existing["w"],
                    existing["h"]
                )

            )


            if(

                overlap >= 0.50

                and

                distance <= max(
                    8,
                    size * 0.45
                )

            ):

                duplicate = True

                break


        if not duplicate:

            kept.append(
                candidate
            )


    cleaned = {

        name: []

        for name
        in detections

    }


    for item in kept:


        cleaned[
            item["name"]
        ].append({

            "x": item["x"],

            "y": item["y"],

            "w": item["w"],

            "h": item["h"],

            "score": item["score"]

        })


    return cleaned


# ============================================================
# 設備模板辨識
# ============================================================

def detect_symbols(
    image
):

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )


    templates =
        load_symbol_templates()


    detections = {}


    for template_data in templates:


        name =
            template_data["name"]


        template =
            template_data["image"]


        th, tw =
            template.shape[:2]


        if(

            th > gray.shape[0]

            or

            tw > gray.shape[1]

        ):

            detections[name] = []

            continue


        result = cv2.matchTemplate(

            gray,

            template,

            cv2.TM_CCOEFF_NORMED

        )


        # 可自行微調
        threshold = 0.62


        ys, xs = np.where(

            result >= threshold

        )


        matches = []


        for y, x in zip(
            ys,
            xs
        ):


            matches.append({

                "x": int(x),

                "y": int(y),

                "w": int(tw),

                "h": int(th),

                "score": float(
                    result[y, x]
                )

            })


        matches =
            remove_duplicates(
                matches
            )


        detections[name] =
            matches


    detections = (
        remove_cross_class_duplicates(
            detections
        )
    )


    return detections


# ============================================================
# 畫紅圈與設備名稱
# ============================================================

def draw_detections(
    image,
    detections
):

    result =
        image.copy()


    for name, matches in detections.items():


        for match in matches:


            x = match["x"]
            y = match["y"]

            w = match["w"]
            h = match["h"]


            cx =
                x + w // 2

            cy =
                y + h // 2


            radius = max(

                10,

                int(
                    max(w, h)
                    * 0.65
                )

            )


            cv2.circle(

                result,

                (
                    cx,
                    cy
                ),

                radius,

                (
                    0,
                    0,
                    255
                ),

                2,

                cv2.LINE_AA

            )


    return result


# ============================================================
# 統計分類
# ============================================================

def count_by_category(
    detections
):

    counts = {}


    for name, matches in detections.items():

        counts[name] = len(matches)


    return counts


# ============================================================
# 分析單一圖面
# ============================================================

def analyze_upload(
    upload,
    page_number
):

    image = load_upload(

        upload,

        page_number

    )


    detections = detect_symbols(
        image
    )


    marked = draw_detections(

        image,

        detections

    )


    counts = count_by_category(
        detections
    )


    total = sum(
        counts.values()
    )


    return {

        "counts": counts,

        "total": total,

        "image": image_to_data_url(
            marked
        )

    }


# ============================================================
# 首頁
# ============================================================

@app.get("/")
def index():

    return render_template_string(
        INDEX_HTML
    )


# ============================================================
# 分析 API
# ============================================================

@app.post("/analyze")
def analyze_api():

    old_file =
        request.files.get("old")

    new_file =
        request.files.get("new")


    if(
        not old_file
        or
        not new_file
    ):

        return jsonify(

            error="請同時選擇原始圖與修改圖。"

        ), 400


    try:


        old_page = int(

            request.form.get(
                "old_page",
                "4"
            )

        )


        new_page = int(

            request.form.get(
                "new_page",
                "3"
            )

        )


        old_result = analyze_upload(

            old_file,

            old_page

        )


        new_result = analyze_upload(

            new_file,

            new_page

        )


        old_counts =
            old_result["counts"]

        new_counts =
            new_result["counts"]


        all_names = sorted(

            set(
                old_counts.keys()
            )

            |

            set(
                new_counts.keys()
            )

        )


        rows = []


        total_added = 0

        total_removed = 0


        for name in all_names:


            old_count =
                old_counts.get(
                    name,
                    0
                )


            new_count =
                new_counts.get(
                    name,
                    0
                )


            difference = (
                new_count
                -
                old_count
            )


            added = max(
                difference,
                0
            )


            removed = max(
                -difference,
                0
            )


            total_added += added

            total_removed += removed


            rows.append({

                "name": name,

                "old_count": old_count,

                "new_count": new_count,

                "added": added,

                "removed": removed,

                "difference": difference

            })


        return jsonify(

            old_total=
                old_result["total"],

            new_total=
                new_result["total"],

            total_added=
                total_added,

            total_removed=
                total_removed,

            old_image=
                old_result["image"],

            new_image=
                new_result["image"],

            rows=rows

        )


    except ValueError as exc:


        return jsonify(

            error=str(exc)

        ), 400


    except Exception as exc:


        app.logger.exception(
            "DrawDiff analyze failed"
        )


        return jsonify(

            error=
                f"辨識失敗：{exc}"

        ), 500


# ============================================================
# CSV 匯出
# ============================================================

@app.post("/export")
def export_csv():

    data =
        request.get_json(
            silent=True
        ) or {}


    rows =
        data.get(
            "rows",
            []
        )


    output =
        io.StringIO()


    writer =
        csv.writer(
            output
        )


    writer.writerow([

        "設備項目",

        "原圖",

        "修改後",

        "增加",

        "減少",

        "淨增減"

    ])


    for row in rows:


        writer.writerow([

            row.get(
                "name",
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

            row.get(
                "added",
                0
            ),

            row.get(
                "removed",
                0
            ),

            row.get(
                "difference",
                0
            )

        ])


    return Response(

        "\ufeff"
        +
        output.getvalue(),

        mimetype=
            "text/csv; charset=utf-8",

        headers={

            "Content-Disposition":

            "attachment; "

            "filename=DrawDiff設備增減統計.csv"

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

        port=port,

        debug=False

    )

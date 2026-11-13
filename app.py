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

RENDER_SCALE = 4.0

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ============================================================
# 設備分類
#
# 左邊：模板檔案名稱
# 右邊：網站顯示名稱
# ============================================================

SYMBOL_CATEGORIES = {

    "馬桶專用插座":
        "馬桶專用插座",

    "抽油煙機插座":
        "抽油煙機插座",

    "雙聯插座_H30_電視牆H80":
        "雙聯插座 H:30cm／電視牆 H:80cm",

    "雙聯插座_H110":
        "雙聯插座 H:110cm",

    "洗碗機專用插座":
        "洗碗機專用插座",

    "微波烤箱專用插座_220V":
        "微波烤箱專用插座 220V",

    "熱水器用插座":
        "熱水器用插座",

    "廚房櫥櫃燈電源出線口":
        "廚房櫥櫃燈電源出線口",

    "暖風乾燥機電源出口":
        "暖風乾燥機電源出口",

    "紅色緊急插座_弱電箱":
        "紅色緊急插座（弱電箱）",

    "紅色緊急插座_冰箱_電視牆":
        "紅色緊急插座（冰箱／電視牆）",

    "冷氣電源出口":
        "冷氣電源出口",

    "專迴_220V":
        "專迴 220V",

    "專迴_110V":
        "專迴 110V",

    "電話_資訊插座":
        "電話、資訊插座",

    "電視_資訊插座":
        "電視、資訊插座",

    "資訊插座":
        "資訊插座"

}


# ============================================================
# 網頁
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

<title>DrawDiff｜設備增減辨識</title>


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
    margin:0 auto;
}


header{
    max-width:1500px;
    margin:0 auto 20px;
}


h1{
    font-size:26px;
    margin:0 0 8px;
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

    box-shadow:
        0 2px 8px
        rgba(20,40,55,.04);

}


h2{

    font-size:20px;

    margin:
        0
        0
        18px;

}


.upload-grid{

    display:grid;

    grid-template-columns:
        1fr
        1fr;

    gap:16px;

}


.upload-box{

    border:
        1px dashed
        #aabac5;

    background:#fbfcfd;

    padding:18px;

    border-radius:10px;

}


.upload-box strong{

    display:block;

    margin-bottom:8px;

}


input[type=file]{

    display:block;

    width:100%;

    margin-top:10px;

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

    width:90px;

    padding:
        8px
        9px;

    border:
        1px solid
        #cbd6dd;

    border-radius:8px;

    font-size:15px;

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

    padding:
        11px
        18px;

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

    padding:
        10px
        15px;

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


.main-result{

    display:grid;

    grid-template-columns:
        repeat(3,1fr);

    gap:12px;

    margin:
        10px
        0
        20px;

}


.metric{

    border:
        1px solid
        #e0e7eb;

    border-radius:10px;

    padding:14px;

}


.metric .label{

    font-size:13px;

    color:#687987;

}


.metric .value{

    font-size:28px;

    font-weight:700;

    margin-top:4px;

}


.compare-grid{

    display:grid;

    grid-template-columns:
        1fr
        1fr;

    gap:16px;

}


.image-panel{

    border:
        1px solid
        #d7e0e6;

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

    background:#fff;

    border:
        1px solid
        #e5eaee;

}


.method{

    font-size:12px;

    color:#667784;

    margin-top:8px;

    line-height:1.45;

}


table{

    border-collapse:collapse;

    width:100%;

    margin-top:20px;

}


th,
td{

    border-bottom:
        1px solid
        #e3e9ed;

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

    font-weight:bold;

}


.minus-text{

    color:#b42318;

    font-weight:bold;

}


.total-row{

    background:#f4f7f8;

    font-weight:bold;

}


.note{

    color:#687987;

    font-size:13px;

    margin-top:12px;

    line-height:1.5;

}


@media(max-width:850px){

    body{
        padding:12px;
    }

    .upload-grid,
    .compare-grid{
        grid-template-columns:1fr;
    }

    .main-result{
        grid-template-columns:
            1fr
            1fr;
    }

    .card{
        padding:16px;
    }

}

</style>

</head>


<body>


<header>

<h1>DrawDiff｜設備逐項增減辨識</h1>

<div class="sub">

上傳原始圖與設計師修改圖，
自動依設備種類分別統計，
並計算每一個設備項目的增加與減少。

</div>

</header>


<main>


<section class="card">


<h2>上傳圖面</h2>


<form id="uploadForm">


<div class="upload-grid">


<div class="upload-box">

<strong>原始圖</strong>

<div class="sub">
預設讀取 PDF 第 4 頁
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

<span>
圖片檔會忽略
</span>

</div>

</div>


<div class="upload-box">

<strong>設計師修改圖</strong>

<div class="sub">
預設讀取 PDF 第 3 頁
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

<span>
圖片檔會忽略
</span>

</div>

</div>


</div>


<div class="actions">

<button
    class="primary"
    id="analyzeButton"
    type="submit"
>

開始辨識設備增減

</button>


<button
    class="secondary"
    id="exportButton"
    type="button"
    disabled
>

匯出逐項增減表 CSV

</button>


</div>


</form>


<div id="status"></div>


</section>


<section
    class="card"
    id="result"
>


<h2>設備逐項增減結果</h2>


<div
    class="summary"
    id="summary"
></div>


<div class="main-result">


<div class="metric">

<div class="label">
原始圖設備總數
</div>

<div
    class="value"
    id="oldCount"
>
—
</div>

</div>


<div class="metric">

<div class="label">
修改後設備總數
</div>

<div
    class="value"
    id="newCount"
>
—
</div>

</div>


<div class="metric">

<div class="label">
淨增減
</div>

<div
    class="value"
    id="netCount"
>
—
</div>

</div>


</div>


<div class="compare-grid">


<div class="image-panel">

<div class="image-title">

原始圖
（紅圈為辨識到的設備）

</div>

<img
    id="oldResult"
    alt="原始圖辨識結果"
>

<div
    class="method"
    id="oldMethod"
></div>

</div>


<div class="image-panel">

<div class="image-title">

修改後圖
（紅圈為辨識到的設備）

</div>

<img
    id="newResult"
    alt="修改圖辨識結果"
>

<div
    class="method"
    id="newMethod"
></div>

</div>


</div>


<table>


<thead>

<tr>

<th>
設備項目
</th>

<th>
原圖
</th>

<th>
修改後
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

    async (event) => {

        event.preventDefault();


        const oldFile =
            document.getElementById(
                'oldFile'
            ).files[0];


        const newFile =
            document.getElementById(
                'newFile'
            ).files[0];


        if (
            !oldFile
            ||
            !newFile
        ) {

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
            '正在讀取圖面並逐項辨識設備，請稍候。';


        statusEl.className =
            '';


        try {


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


            if (
                !response.ok
            ) {

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
                .getElementById('oldCount')
                .textContent =
                    data.old_count;


            document
                .getElementById('newCount')
                .textContent =
                    data.new_count;


            const net =
                data.difference;


            document
                .getElementById('netCount')
                .textContent =
                    (
                        net > 0
                        ? '+'
                        : ''
                    )
                    +
                    net;


            const summary =
                document.getElementById(
                    'summary'
                );


            summary.innerHTML =
                '';


            [

                [
                    '原圖 '
                    +
                    data.old_count
                    +
                    ' 個',
                    ''
                ],

                [
                    '修改後 '
                    +
                    data.new_count
                    +
                    ' 個',
                    ''
                ],

                [
                    '增加 '
                    +
                    data.added_count
                    +
                    ' 個',
                    'plus'
                ],

                [
                    '減少 '
                    +
                    data.removed_count
                    +
                    ' 個',
                    'minus'
                ]

            ].forEach(

                ([label, cls]) => {

                    const d =
                        document.createElement(
                            'div'
                        );


                    d.className =
                        'pill '
                        +
                        cls;


                    d.textContent =
                        label;


                    summary.appendChild(
                        d
                    );

                }

            );


            document
                .getElementById('oldResult')
                .src =
                    data.old_image;


            document
                .getElementById('newResult')
                .src =
                    data.new_image;


            document
                .getElementById('oldMethod')
                .textContent =
                    data.old_method;


            document
                .getElementById('newMethod')
                .textContent =
                    data.new_method;


            const body =
                document.getElementById(
                    'resultRows'
                );


            body.innerHTML =
                '';


            latestRows =
                data.rows;


            data.rows.forEach(

                row => {


                    const tr =
                        document.createElement(
                            'tr'
                        );


                    const diff =
                        row.difference;


                    const netText =
                        (
                            diff > 0
                            ? '+'
                            : ''
                        )
                        +
                        diff;


                    const values = [

                        row.name,

                        row.old_count,

                        row.new_count,

                        row.added,

                        row.removed,

                        netText

                    ];


                    values.forEach(

                        (value, index) => {

                            const td =
                                document.createElement(
                                    'td'
                                );


                            td.textContent =
                                value;


                            if (
                                index === 3
                                &&
                                row.added > 0
                            ) {

                                td.className =
                                    'plus-text';

                            }


                            if (
                                index === 4
                                &&
                                row.removed > 0
                            ) {

                                td.className =
                                    'minus-text';

                            }


                            if (
                                index === 5
                                &&
                                diff > 0
                            ) {

                                td.className =
                                    'plus-text';

                            }


                            if (
                                index === 5
                                &&
                                diff < 0
                            ) {

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


            const total =
                document.createElement(
                    'tr'
                );


            total.className =
                'total-row';


            const totalValues = [

                '合計',

                data.old_count,

                data.new_count,

                data.added_count,

                data.removed_count,

                (
                    net > 0
                    ? '+'
                    : ''
                )
                +
                net

            ];


            totalValues.forEach(

                value => {

                    const td =
                        document.createElement(
                            'td'
                        );


                    td.textContent =
                        value;


                    total.appendChild(
                        td
                    );

                }

            );


            body.appendChild(
                total
            );


            document
                .getElementById('resultNote')
                .textContent =
                    '每一列為個別設備種類的辨識結果。紅圈為實際辨識位置；請先核對紅圈是否正確，再作為客變追加減帳資料。';


            exportButton.disabled =
                false;


            statusEl.textContent =
                '辨識完成。請先核對左右圖面的紅圈位置及逐項數量。';


            document
                .getElementById('result')
                .scrollIntoView({

                    behavior:'smooth',

                    block:'start'

                });


        }

        catch(err) {


            statusEl.textContent =
                err.message;


            statusEl.className =
                'error';

        }

        finally {


            analyzeButton.disabled =
                false;

        }

    }

);


exportButton.addEventListener(

    'click',

    async () => {


        if (
            !latestRows.length
        ) {

            return;

        }


        try {


            const response =
                await fetch(

                    '/export',

                    {

                        method:'POST',

                        headers:{

                            'Content-Type':
                                'application/json'

                        },

                        body:
                            JSON.stringify({

                                rows:
                                    latestRows

                            })

                    }

                );


            if (
                !response.ok
            ) {

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
                'DrawDiff設備逐項增減統計.csv';


            document
                .body
                .appendChild(a);


            a.click();


            a.remove();


            URL.revokeObjectURL(
                url
            );


        }

        catch(err) {


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
# 取得副檔名
# ============================================================

def filename_ext(filename):

    return Path(
        filename or ""
    ).suffix.lower().lstrip(".")


# ============================================================
# 圖片轉 Base64
# ============================================================

def jpeg_data_url(
    image_bgr,
    max_side=2200,
    quality=88
):

    h, w =
        image_bgr.shape[:2]


    scale = min(
        1.0,
        max_side / max(h, w)
    )


    if scale < 1.0:

        image_bgr = cv2.resize(

            image_bgr,

            (
                int(w * scale),
                int(h * scale)
            ),

            interpolation=
                cv2.INTER_AREA

        )


    ok, buffer =
        cv2.imencode(

            ".jpg",

            image_bgr,

            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                quality
            ]

        )


    if not ok:

        raise RuntimeError(
            "結果圖片編碼失敗"
        )


    encoded =
        base64.b64encode(
            buffer.tobytes()
        ).decode("ascii")


    return (
        "data:image/jpeg;base64,"
        +
        encoded
    )


# ============================================================
# 讀取設備模板
# ============================================================

def load_symbol_templates():

    templates = []


    if not SYMBOL_DIR.exists():

        return templates


    for file_path in SYMBOL_DIR.iterdir():


        if not file_path.is_file():

            continue


        extension =
            file_path.suffix.lower()


        if extension not in {

            ".png",
            ".jpg",
            ".jpeg",
            ".bmp"

        }:

            continue


        symbol_id =
            file_path.stem


        if symbol_id not in SYMBOL_CATEGORIES:

            continue


        template =
            cv2.imread(

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

            "id":
                symbol_id,

            "name":
                SYMBOL_CATEGORIES[
                    symbol_id
                ],

            "image":
                template

        })


    return templates


# ============================================================
# IoU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1 =
        box_a["x"]


    ay1 =
        box_a["y"]


    ax2 =
        ax1 + box_a["w"]


    ay2 =
        ay1 + box_a["h"]


    bx1 =
        box_b["x"]


    by1 =
        box_b["y"]


    bx2 =
        bx1 + box_b["w"]


    by2 =
        by1 + box_b["h"]


    x1 =
        max(ax1, bx1)


    y1 =
        max(ay1, by1)


    x2 =
        min(ax2, bx2)


    y2 =
        min(ay2, by2)


    intersection_w =
        max(0, x2 - x1)


    intersection_h =
        max(0, y2 - y1)


    intersection =
        (
            intersection_w
            *
            intersection_h
        )


    area_a =
        (
            (ax2 - ax1)
            *
            (ay2 - ay1)
        )


    area_b =
        (
            (bx2 - bx1)
            *
            (by2 - by1)
        )


    union =
        (
            area_a
            +
            area_b
            -
            intersection
        )


    if union <= 0:

        return 0.0


    return (
        intersection
        /
        union
    )


# ============================================================
# 同一設備去除重複
# ============================================================

def remove_duplicate_matches(matches):


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


            cx =
                candidate["x"]
                +
                candidate["w"] / 2


            cy =
                candidate["y"]
                +
                candidate["h"] / 2


            ex =
                existing["x"]
                +
                existing["w"] / 2


            ey =
                existing["y"]
                +
                existing["h"] / 2


            distance = (

                (
                    (cx - ex) ** 2
                    +
                    (cy - ey) ** 2
                )

                ** 0.5

            )


            size =
                min(

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
# 不同設備之間去除重複
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


    all_items.sort(

        key=lambda item:
            item["score"],

        reverse=True

    )


    kept = []


    for candidate in all_items:


        duplicate = False


        cx =
            candidate["x"]
            +
            candidate["w"] / 2


        cy =
            candidate["y"]
            +
            candidate["h"] / 2


        for existing in kept:


            ex =
                existing["x"]
                +
                existing["w"] / 2


            ey =
                existing["y"]
                +
                existing["h"] / 2


            distance = (

                (
                    (cx - ex) ** 2
                    +
                    (cy - ey) ** 2
                )

                ** 0.5

            )


            overlap =
                calculate_iou(

                    candidate,

                    existing

                )


            candidate_size =
                max(

                    candidate["w"],

                    candidate["h"]

                )


            existing_size =
                max(

                    existing["w"],

                    existing["h"]

                )


            min_size =
                min(

                    candidate_size,

                    existing_size

                )


            # 同一位置被不同模板辨識到
            if (

                overlap >= 0.45

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

        symbol_id: []

        for symbol_id
        in detections

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
# 多尺寸模板辨識
#
# 同一設備可能因 PDF 縮放而大小不同
# ============================================================

def detect_symbols(image_bgr):


    if image_bgr is None:

        return {}


    gray =
        cv2.cvtColor(

            image_bgr,

            cv2.COLOR_BGR2GRAY

        )


    templates =
        load_symbol_templates()


    if not templates:

        raise ValueError(

            "找不到設備模板。請確認 "
            "static/symbols 資料夾內已有設備圖案。"

        )


    detections = {

        symbol_id: []

        for symbol_id
        in SYMBOL_CATEGORIES

    }


    # 多尺寸辨識倍率
    scales = [

        0.75,

        0.85,

        0.95,

        1.00,

        1.05,

        1.15,

        1.25

    ]


    for template_data in templates:


        symbol_id =
            template_data["id"]


        original_template =
            template_data["image"]


        matches = []


        for scale in scales:


            template =
                cv2.resize(

                    original_template,

                    None,

                    fx=scale,

                    fy=scale,

                    interpolation=
                        cv2.INTER_AREA

                )


            template_h =
                template.shape[0]


            template_w =
                template.shape[1]


            if (

                template_h < 5

                or

                template_w < 5

            ):

                continue


            if (

                template_h > gray.shape[0]

                or

                template_w > gray.shape[1]

            ):

                continue


            result =
                cv2.matchTemplate(

                    gray,

                    template,

                    cv2.TM_CCOEFF_NORMED

                )


            # ------------------------------------------------
            # 辨識門檻
            #
            # 如果漏辨識：
            # 可以改成 0.58
            #
            # 如果誤辨識太多：
            # 可以改成 0.68
            # ------------------------------------------------

            threshold = 0.62


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


                matches.append({

                    "x":
                        int(x),

                    "y":
                        int(y),

                    "w":
                        int(template_w),

                    "h":
                        int(template_h),

                    "score":
                        score

                })


        matches =
            remove_duplicate_matches(
                matches
            )


        detections[
            symbol_id
        ] = matches


    detections =
        remove_cross_class_duplicates(
            detections
        )


    return detections


# ============================================================
# 在圖片畫紅圈 + 分類名稱
# ============================================================

def draw_detected_symbols(
    image,
    detections
):


    result =
        image.copy()


    number = 1


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
                        *
                        0.65
                    )

                )


            # =================================================
            # 紅圈
            # =================================================

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

                2,

                cv2.LINE_AA

            )


            # =================================================
            # 編號
            # =================================================

            label =
                str(number)


            label_x =
                max(
                    14,
                    center_x - radius
                )


            label_y =
                max(
                    14,
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

                -1,

                cv2.LINE_AA

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


            number += 1


    return result


# ============================================================
# 計算分類結果
# ============================================================

def count_by_category(detections):


    result = {}


    for symbol_id in SYMBOL_CATEGORIES:


        result[symbol_id] = len(

            detections.get(
                symbol_id,
                []
            )

        )


    return result


# ============================================================
# 分析 PDF
# ============================================================

def analyze_pdf(
    file_bytes,
    page_number
):


    document = None


    try:


        document =
            fitz.open(

                stream=file_bytes,

                filetype="pdf"

            )


        page_index =
            page_number - 1


        if (

            page_index < 0

            or

            page_index >= len(document)

        ):


            raise ValueError(

                f"PDF 只有 {len(document)} 頁，"
                f"指定的第 {page_number} 頁不存在。"

            )


        page =
            document.load_page(
                page_index
            )


        pix =
            page.get_pixmap(

                matrix=
                    fitz.Matrix(

                        RENDER_SCALE,

                        RENDER_SCALE

                    ),

                alpha=False,

                colorspace=
                    fitz.csRGB

            )


        rgb =
            np.frombuffer(

                pix.samples,

                dtype=np.uint8

            ).reshape(

                pix.height,

                pix.width,

                pix.n

            )


        image_bgr =
            cv2.cvtColor(

                rgb,

                cv2.COLOR_RGB2BGR

            )


        detections =
            detect_symbols(
                image_bgr
            )


        marked =
            draw_detected_symbols(

                image_bgr,

                detections

            )


        counts =
            count_by_category(
                detections
            )


        return {

            "image":
                jpeg_data_url(
                    marked
                ),

            "counts":
                counts,

            "method":
                "設備圖案模板辨識"

        }


    finally:


        if document is not None:

            document.close()


# ============================================================
# 分析圖片
# ============================================================

def analyze_image(file_bytes):


    array =
        np.frombuffer(

            file_bytes,

            dtype=np.uint8

        )


    image =
        cv2.imdecode(

            array,

            cv2.IMREAD_COLOR

        )


    if image is None:

        raise ValueError(
            "圖片無法讀取。"
        )


    detections =
        detect_symbols(
            image
        )


    marked =
        draw_detected_symbols(

            image,

            detections

        )


    counts =
        count_by_category(
            detections
        )


    return {

        "image":
            jpeg_data_url(
                marked
            ),

        "counts":
            counts,

        "method":
            "設備圖案模板辨識"

    }


# ============================================================
# 分析上傳檔案
# ============================================================

def analyze_upload(
    upload,
    page_number
):


    filename =
        upload.filename or ""


    ext =
        filename_ext(
            filename
        )


    if ext not in ALLOWED:


        raise ValueError(

            "只支援 PDF、PNG、JPG、JPEG、"
            "BMP、TIF、TIFF。"

        )


    raw =
        upload.read()


    if not raw:


        raise ValueError(

            f"檔案是空的：{filename}"

        )


    if ext == "pdf":


        return analyze_pdf(

            raw,

            page_number

        )


    return analyze_image(
        raw
    )


# ============================================================
# 首頁
# ============================================================

@app.get("/")
def index():

    return render_template_string(
        INDEX_HTML
    )


# ============================================================
# 開始分析
# ============================================================

@app.post("/analyze")
def analyze_api():


    old_file =
        request.files.get("old")


    new_file =
        request.files.get("new")


    if (

        not old_file

        or

        not new_file

    ):


        return jsonify(

            error=
                "請同時選擇原始圖及修改後圖。"

        ), 400


    try:


        old_page =
            int(

                request.form.get(
                    "old_page",
                    "4"
                )

            )


        new_page =
            int(

                request.form.get(
                    "new_page",
                    "3"
                )

            )


        if (

            old_page < 1

            or

            new_page < 1

        ):


            raise ValueError(
                "PDF 頁碼必須從 1 開始。"
            )


        # ====================================================
        # 原圖
        # ====================================================

        old_result =
            analyze_upload(

                old_file,

                old_page

            )


        # ====================================================
        # 修改圖
        # ====================================================

        new_result =
            analyze_upload(

                new_file,

                new_page

            )


        old_counts =
            old_result["counts"]


        new_counts =
            new_result["counts"]


        # ====================================================
        # 建立逐項表格
        # ====================================================

        rows = []


        total_old = 0

        total_new = 0


        total_added = 0

        total_removed = 0


        for symbol_id, display_name in (
            SYMBOL_CATEGORIES.items()
        ):


            old_count =
                old_counts.get(
                    symbol_id,
                    0
                )


            new_count =
                new_counts.get(
                    symbol_id,
                    0
                )


            difference =
                new_count - old_count


            added =
                max(
                    difference,
                    0
                )


            removed =
                max(
                    -difference,
                    0
                )


            rows.append({

                "name":
                    display_name,

                "old_count":
                    old_count,

                "new_count":
                    new_count,

                "added":
                    added,

                "removed":
                    removed,

                "difference":
                    difference

            })


            total_old += old_count

            total_new += new_count

            total_added += added

            total_removed += removed


        difference =
            total_new - total_old


        return jsonify(


            old_count=
                total_old,


            new_count=
                total_new,


            difference=
                difference,


            added_count=
                total_added,


            removed_count=
                total_removed,


            old_image=
                old_result["image"],


            new_image=
                new_result["image"],


            old_method=
                old_result["method"],


            new_method=
                new_result["method"],


            rows=
                rows

        )


    except ValueError as exc:


        return jsonify(

            error=str(exc)

        ), 400


    except Exception as exc:


        app.logger.exception(
            "analysis failed"
        )


        return jsonify(

            error=
                f"辨識失敗：{exc}"

        ), 500


# ============================================================
# 匯出 CSV
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

        "原圖數量",

        "修改後數量",

        "增加",

        "減少",

        "淨增減"

    ])


    total_old = 0

    total_new = 0

    total_added = 0

    total_removed = 0


    for row in rows:


        old_count =
            int(
                row.get(
                    "old_count",
                    0
                )
            )


        new_count =
            int(
                row.get(
                    "new_count",
                    0
                )
            )


        diff =
            new_count - old_count


        added =
            max(
                diff,
                0
            )


        removed =
            max(
                -diff,
                0
            )


        writer.writerow([

            row.get(
                "name",
                ""
            ),

            old_count,

            new_count,

            added,

            removed,

            diff

        ])


        total_old += old_count

        total_new += new_count

        total_added += added

        total_removed += removed


    writer.writerow([

        "合計",

        total_old,

        total_new,

        total_added,

        total_removed,

        total_new - total_old

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

                "filename=DrawDiff設備逐項增減統計.csv"

        }

    )


# ============================================================
# Render 啟動
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

        debug=False

    )

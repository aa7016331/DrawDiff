import os
import csv
import io
import base64
from pathlib import Path
from collections import Counter

import cv2
import fitz
import numpy as np
from flask import Flask, render_template_string, request, jsonify, Response


BASE = Path(__file__).resolve().parent
SYMBOL_DIR = BASE / "static" / "symbols"

ALLOWED = {"pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# 0～12 號符號固定分類名稱
SYMBOL_NAMES = {
    0: "住戶開關箱與弱電箱整合",
    1: "馬桶專用插座 H:30cm",
    2: "抽油煙機插座 H:天花板內",
    3: "雙聯插座 H:30cm／電視牆 H:80cm",
    4: "雙聯插座 H:110cm",
    5: "洗碗機專用插座 H:45cm",
    6: "微波烤箱專用插座 220V",
    7: "洗衣機用插座（防雨型）H:120cm",
    8: "廚房橫櫃燈用電源出線口 H:170cm",
    9: "暖風乾燥機電源出口 220V",
    10: "紅色緊急插座（弱電箱）",
    11: "紅色緊急插座（冰箱／電視牆）",
    12: "冷氣電源出口 220V",
}

# 分類參數：精準優先，無法清楚辨認時保留未分類。
CLASSIFY_CANVAS_SIZE = 176
CLASSIFY_EDGE_TOLERANCE = 2.5
CLASSIFY_SCALE_VARIANTS = (0.92, 1.0, 1.08)
CLASSIFY_CODE_MIN_SCORE = 0.70
CLASSIFY_CODE_MIN_MARGIN = 0.03
CLASSIFY_BODY_ONLY_MIN_SCORE = 0.86
CLASSIFY_BODY_ONLY_MIN_MARGIN = 0.10

# 範本內部的英文代碼位置（以各 PNG 原始像素座標表示）。
# 這些是符號圖形的一部分，使用影像特徵比對，不使用 OCR。
GLYPH_BOXES = {
    2: (40, 17, 47, 29),   # F
    4: (39, 14, 47, 27),   # H
    5: (39, 16, 47, 28),   # G
    6: (38, 17, 46, 28),   # K
    7: (38, 18, 48, 29),   # W
    9: (39, 15, 46, 28),   # H
    10: (40, 30, 48, 43),  # C
    11: (39, 38, 46, 47),  # E
    12: (39, 17, 46, 28),  # H
}
INTERNAL_GLYPH_BOXES = {
    8: (21, 8, 32, 25),    # J，位於圓形內部
}
CODE_SYMBOL_IDS = (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)
BODY_ONLY_SYMBOL_IDS = (1, 3)

# 代碼字形的跨類別驗證。H 由 4／9／12 共用；其他代碼各自獨立。
# 這一步不是 OCR，而是把紅圈候選中代碼所在固定區域的形狀，
# 與 13 張官方範本內的英文字形特徵逐一比對，防止把 H 誤當 W 等情況。
GLYPH_GROUPS = {
    "F": (2,),
    "H": (4, 9, 12),
    "G": (5,),
    "K": (6,),
    "W": (7,),
    "J": (8,),
    "C": (10,),
    "E": (11,),
}
SYMBOL_GLYPH_GROUP = {
    symbol_id: glyph_group
    for glyph_group, symbol_ids in GLYPH_GROUPS.items()
    for symbol_id in symbol_ids
}
# 若另一種字母的分數明顯高於候選所屬代碼，該候選不能通過。
# G/C 筆畫相近，採較寬容的衝突門檻，再由圓內結構與整體分數把關。
GLYPH_CODE_CONFLICT_MARGIN = {
    2: 0.035, 4: 0.010, 5: 0.070, 6: 0.030, 7: 0.025,
    8: 0.035, 9: 0.010, 10: 0.045, 11: 0.030, 12: 0.010,
}

# 範本中的圓心與半徑（原始範本像素座標）。
# 0 號為長方形箱體符號，並非圓形符號，不交由目前的圓形紅圈候選分類。
TEMPLATE_CIRCLE_GEOMETRY = {
    1: (26.0, 14.0, 13.0),
    2: (26.0, 14.0, 13.0),
    3: (26.0, 14.0, 13.0),
    4: (26.0, 14.0, 13.0),
    5: (26.0, 14.0, 13.0),

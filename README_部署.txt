DrawDiff 正式部署版

這是「網站伺服器版」，使用者不需要安裝 Python 或 CMD。

部署後使用方式：
1. 打開公開網址
2. 上傳原始圖
3. 上傳修改後圖
4. 按開始自動核對
5. 系統在伺服器端辨識 13 種固定圖例並計算增減

可部署到 Render、Railway、Fly.io 或任何支援 Docker/Python 的雲端平台。

Render 基本設定：
- Build Command: pip install -r requirements.txt
- Start Command: gunicorn app:app
- Python: 3.11+
或直接使用 Dockerfile。

重要：目前辨識核心使用多尺度模板比對。若實際施工圖的符號比例、解析度或線條風格與目前圖例差異很大，需要用實際圖面調整 threshold/scale，才能提高準確度。

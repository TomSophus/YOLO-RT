
# YOLO-RT: Real-time Object Detection with YOLOv8

**YOLO-RT** は、Ultralytics社が提供する YOLOv8 モデルを使用し、PCのWebカメラからリアルタイムに人物や物体を検出・表示する Python アプリケーションです。OpenCV を用いたビジュアライゼーションにより、検出結果やFPSなどの統計情報をオーバーレイ表示できます。

---

## 🎥 デモ

![demo](demo.gif)  
※ デモ動画（例：`demo.mp4`）は `.gitignore` に含めるか、[Git LFS](https://git-lfs.com/) で管理することを推奨します。

---

## 📁 ディレクトリ構成

```
YOLO-RT/
├── src/
│   ├── camera_handler.py       # カメラ制御（cv2.VideoCaptureのラッパー）
│   ├── model_handler.py        # YOLOモデルの読み込みと推論処理
│   ├── detection_utils.py      # 可視化や検出後処理（描画・モザイクなど）
│   └── config.py               # 設定ファイル（モデル・カメラ・表示など）
├── notebooks/
│   └── main_detection.ipynb    # ノートブック形式での動作確認スクリプト
├── requirements.txt            # 使用ライブラリ一覧
├── .gitignore                  # Git追跡対象外のファイル定義
└── README.md
```

---

## ⚙️ セットアップ方法

### 1. 仮想環境の作成（任意）

```bash
python -m venv venv
source venv/bin/activate        # Windowsでは venv\Scripts\activate
```

### 2. 必要ライブラリのインストール

```bash
pip install -r requirements.txt
```

主な依存ライブラリ:
- `ultralytics`
- `opencv-python`
- `numpy`
- `torch`

---

## 🚀 実行手順

### 方法1: Jupyter Notebookで実行

```bash
jupyter notebook notebooks/main_detection.ipynb
```

### 方法2: `.py`ファイルに変換して実行（例: `main_detection.py`）

```bash
python src/main_detection.py
```

※ ウィンドウは `ESC` キーで終了可能です。

---

## 🔧 主な設定とカスタマイズ

すべての設定は `src/config.py` にて集中管理されています。

### 例: 使用モデルの変更
```python
DEFAULT_MODEL = "yolov8n"
```

### 例: カメラ設定の変更
```python
CAMERA_CONFIG = {
    "width": 1920,
    "height": 1080,
    "fps": 30,
    ...
}
```

### 例: 日本語ラベルの使用
```python
model = ModelHandler(use_japanese_labels=True)
```

---

## ✨ 特徴

- ✔️ YOLOv8（nano〜x）および YOLOv11 シリーズに対応
- ✔️ 高速・軽量なリアルタイム処理
- ✔️ FPS、検出数、モデル名のオーバーレイ表示
- ✔️ モザイク処理や検出フィルタ機能搭載
- ✔️ クラス名の日本語・英語切替機能

---

## 📚 使用ライブラリ

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)

---

## 📝 ライセンス

このプロジェクトで使用するYOLOv8nは **AGPL-3.0ライセンス** の下で提供されています。

---

## 🙋‍♂️ 貢献・連絡

バグ報告、機能改善の提案、プルリクエストは大歓迎です！  
何かあればお気軽に Issue を立ててください。

---

**© 2025 TomSophus**

"""
Configuration file for YOLO real-time object detection
YOLO リアルタイム物体検出アプリケーションの設定ファイル
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ============================================================================
# パス設定
# ============================================================================

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# モデル保存ディレクトリ
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 出力ディレクトリ
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ログディレクトリ
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# モデル設定
# ============================================================================

# 使用可能なYOLOモデル
AVAILABLE_MODELS = {
    "yolov8n": "yolov8n.pt",      # nano - 最軽量
    "yolov8s": "yolov8s.pt",      # small
    "yolov8m": "yolov8m.pt",      # medium
    "yolov8l": "yolov8l.pt",      # large
    "yolov8x": "yolov8x.pt",      # extra large - 最高精度
    "yolov11n": "yolo11n.pt",     # YOLO11 nano
    "yolov11s": "yolo11s.pt",     # YOLO11 small
    "yolov11m": "yolo11m.pt",     # YOLO11 medium
    "yolov11l": "yolo11l.pt",     # YOLO11 large
    "yolov11x": "yolo11x.pt",     # YOLO11 extra large
}

# デフォルトモデル
DEFAULT_MODEL = "yolov8n"

# モデル設定
MODEL_CONFIG = {
    "confidence_threshold": 0.5,   # 信頼度閾値
    "iou_threshold": 0.45,         # IoU閾値（NMS用）
    "max_detections": 300,         # 最大検出数
    "device": "auto",              # デバイス設定 ("auto", "cpu", "cuda", "mps")
}

# ============================================================================
# カメラ設定
# ============================================================================

CAMERA_CONFIG = {
    "camera_id": 0,                # カメラID（通常は0がデフォルト）
    "width": 1920,                  # 解像度（幅）　　　4K：3840　FHD：1920
    "height": 1080,                 # 解像度（高さ）　　4K：2160　FHD：1080
    "fps": 30,                     # フレームレート
    "buffer_size": 1,              # バッファサイズ（低遅延のため）
    "use_native_resolution": True  # カメラ映像をそのまま使う → TRUE 
                                   # サイズを指定する → FALSE
}

# ============================================================================
# 表示設定
# ============================================================================

# 描画設定
VISUALIZATION_CONFIG = {
    "bbox_thickness": 2,           # バウンディングボックスの線の太さ
    "font_scale": 0.6,             # フォントサイズ
    "font_thickness": 2,           # フォントの太さ
    "label_padding": 5,            # ラベルのパディング
    "show_confidence": True,       # 信頼度表示
    "show_class_name": True,       # クラス名表示
    "show_fps": True,              # FPS表示
    "show_detection_count": True,  # 検出数表示
    "show_model_name": True,       # モデル名表示
}

# 色設定
COLOR_CONFIG = {
    "use_random_colors": True,     # ランダム色使用
    "color_saturation": 0.8,       # 色の彩度 (0.0-1.0)
    "color_value": 0.8,            # 色の明度 (0.0-1.0)
    "text_color": (255, 255, 255), # テキスト色 (BGR)
    "info_bg_color": (0, 0, 0),    # 情報表示背景色 (BGR)
}

# ============================================================================
# フィルタリング設定
# ============================================================================

FILTER_CONFIG = {
    "confidence_threshold": 0.5,   # 信頼度フィルタ
    "min_bbox_area": 100,          # 最小バウンディングボックス面積
    "max_bbox_area": None,         # 最大バウンディングボックス面積
    "allowed_classes": None,       # 許可するクラス（Noneで全許可）
}

# ============================================================================
# クラス名設定
# ============================================================================

# COCO データセットのクラス名（80クラス）
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# 日本語クラス名（オプション）
COCO_CLASS_NAMES_JP = [
    '人', '自転車', '車', 'バイク', '飛行機', 'バス', '電車', 'トラック',
    'ボート', '信号機', '消火栓', '停止標識', 'パーキングメーター', 'ベンチ',
    '鳥', '猫', '犬', '馬', '羊', '牛', '象', '熊', 'シマウマ',
    'キリン', 'リュック', '傘', 'ハンドバッグ', 'ネクタイ', 'スーツケース', 'フリスビー',
    'スキー', 'スノーボード', 'スポーツボール', '凧', '野球バット', '野球グローブ',
    'スケートボード', 'サーフボード', 'テニスラケット', 'ボトル', 'ワイングラス', 'カップ',
    'フォーク', 'ナイフ', 'スプーン', 'ボウル', 'バナナ', 'りんご', 'サンドイッチ', 'オレンジ',
    'ブロッコリー', 'にんじん', 'ホットドッグ', 'ピザ', 'ドーナツ', 'ケーキ', '椅子', 'ソファ',
    '観葉植物', 'ベッド', 'ダイニングテーブル', 'トイレ', 'テレビ', 'ノートPC', 'マウス',
    'リモコン', 'キーボード', '携帯電話', '電子レンジ', 'オーブン', 'トースター', 'シンク',
    '冷蔵庫', '本', '時計', '花瓶', 'はさみ', 'テディベア', 'ドライヤー', '歯ブラシ'
]

# 使用するクラス名（デフォルトは英語）
DEFAULT_CLASS_NAMES = COCO_CLASS_NAMES

# ============================================================================
# パフォーマンス設定
# ============================================================================

PERFORMANCE_CONFIG = {
    "enable_gpu": True,            # GPU使用
    "enable_half_precision": False, # 半精度推論（GPU必要）
    "enable_tensorrt": False,      # TensorRT最適化
    "thread_count": None,          # スレッド数（Noneで自動）
    "batch_size": 1,               # バッチサイズ
}

# ============================================================================
# ログ設定
# ============================================================================

LOGGING_CONFIG = {
    "log_level": "INFO",           # ログレベル
    "log_to_file": True,           # ファイル出力
    "log_to_console": True,        # コンソール出力
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "detection.log",
}

# ============================================================================
# 出力・保存設定
# ============================================================================

OUTPUT_CONFIG = {
    "save_results": False,         # 結果保存
    "save_video": False,           # 動画保存
    "output_format": "mp4",        # 出力形式
    "output_fps": 30,              # 出力FPS
    "output_quality": 80,          # 出力品質 (1-100)
}

# ============================================================================
# ウィンドウ設定
# ============================================================================

WINDOW_CONFIG = {
    "window_name": "YOLO Real-time Detection",
    "window_width": 800,
    "window_height": 600,
    "resizable": True,
    "show_fullscreen": False,
}

# ============================================================================
# 開発・デバッグ設定
# ============================================================================

DEBUG_CONFIG = {
    "debug_mode": False,           # デバッグモード
    "show_processing_time": False, # 処理時間表示
    "save_debug_images": False,    # デバッグ画像保存
    "verbose_logging": False,      # 詳細ログ
}

# ============================================================================
# 設定取得関数
# ============================================================================

def get_model_path(model_name: str) -> str:
    """モデルパスを取得"""
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")

def get_class_names(language: str = "en") -> List[str]:
    """指定言語のクラス名を取得"""
    if language == "jp" or language == "ja":
        return COCO_CLASS_NAMES_JP
    else:
        return COCO_CLASS_NAMES

def validate_config() -> bool:
    """設定の妥当性をチェック"""
    try:
        # 必須ディレクトリの存在確認
        for dir_path in [MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # 設定値の範囲チェック
        assert 0.0 <= MODEL_CONFIG["confidence_threshold"] <= 1.0
        assert 0.0 <= MODEL_CONFIG["iou_threshold"] <= 1.0
        assert CAMERA_CONFIG["width"] > 0
        assert CAMERA_CONFIG["height"] > 0
        assert CAMERA_CONFIG["fps"] > 0
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

def update_config(section: str, key: str, value: Any) -> None:
    """設定値を動的に更新"""
    config_dict = globals()
    if section in config_dict and isinstance(config_dict[section], dict):
        config_dict[section][key] = value
        print(f"Updated {section}[{key}] = {value}")
    else:
        print(f"Invalid section: {section}")

def get_config_summary() -> Dict[str, Any]:
    """設定の要約を取得"""
    return {
        "model": DEFAULT_MODEL,
        "confidence_threshold": MODEL_CONFIG["confidence_threshold"],
        "camera_resolution": f"{CAMERA_CONFIG['width']}x{CAMERA_CONFIG['height']}",
        "camera_fps": CAMERA_CONFIG["fps"],
        "device": MODEL_CONFIG["device"],
        "debug_mode": DEBUG_CONFIG["debug_mode"],
    }

# ============================================================================
# 初期化時の設定確認
# ============================================================================

if __name__ == "__main__":
    print("=== YOLO Detection Configuration ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Available Models: {list(AVAILABLE_MODELS.keys())}")
    print(f"Default Model: {DEFAULT_MODEL}")
    
    # 設定検証
    if validate_config():
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
    
    # 設定要約表示
    summary = get_config_summary()
    print("\n=== Configuration Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
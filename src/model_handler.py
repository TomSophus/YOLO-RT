"""
Model Handler for YOLO Real-time Object Detection
YOLOモデルの読み込み、推論、結果処理を行うモジュール（config.py連携）
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import cv2

from config import (
    MODEL_CONFIG, DEFAULT_MODEL, get_model_path, get_class_names
)

class ModelHandler:
    """
    YOLOモデルのロード・推論・出力パースをまとめて管理するクラス
    """

    def __init__(self, 
                 model_name: str = DEFAULT_MODEL,
                 use_japanese_labels: bool = False):
        """
        モデルハンドラーの初期化（config.pyから設定値を取得）

        Args:
            model_name (str): 使用するYOLOモデル名（例: "yolov8n"）
            use_japanese_labels (bool): 日本語クラス名を使うか（デフォルトは英語）
        """
        self.model_name = model_name
        self.model_path = get_model_path(model_name)
        self.device = self._setup_device(MODEL_CONFIG["device"])
        self.conf_threshold = MODEL_CONFIG["confidence_threshold"]
        self.iou_threshold = MODEL_CONFIG["iou_threshold"]
        self.max_detections = MODEL_CONFIG["max_detections"]
        self.use_japanese_labels = use_japanese_labels

        self.model = None  # YOLOインスタンス
        self.class_names = []
        self.is_loaded = False

        self.inference_times = []       # 推論時間履歴
        self.total_detections = 0       # 累積検出数

    def _setup_device(self, device: str) -> str:
        """
        使用するデバイス（CPU/GPU/MPS）を自動または指定で決定
        """
        if device == "auto":
            if torch.cuda.is_available():
                print("GPU使用:", torch.cuda.get_device_name(0))
                return "cuda"
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                print("MPS (Apple GPU) 使用")
                return "mps"
            else:
                print("CPU使用")
                return "cpu"
        else:
            print("指定デバイス使用:", device)
            return device

    def load_model(self) -> bool:
        """
        YOLOモデルの読み込みと初期化

        Returns:
            bool: 読み込み成功ならTrue、失敗時False
        """
        try:
            print(f"モデル読み込み中: {self.model_path}")
            self.model = YOLO(self.model_path)

            if self.device != "cpu":
                self.model.to(self.device)

            # クラス名取得（日本語 or 英語）
            self.class_names = get_class_names("jp" if self.use_japanese_labels else "en")
            print(f"モデル読み込み完了 - デバイス: {self.device}, クラス数: {len(self.class_names)}")
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False

    def predict(self, frame: np.ndarray, verbose: bool = False) -> Optional[Dict]:
        """
        推論を実行し、結果を辞書形式で返す

        Args:
            frame (np.ndarray): 入力画像（BGR形式）
            verbose (bool): Ultralyticsの詳細出力を有効にするか

        Returns:
            Optional[Dict]: 推論結果と統計情報（失敗時はNone）
        """
        if not self.is_loaded or self.model is None:
            print("モデルが未読み込みです")
            return None

        try:
            start_time = time.time()

            # Ultralytics YOLOv8にNumPy画像をそのまま渡す
            h, w = frame.shape[:2]
            results = self.model(
                frame,
                imgsz=max(h, w),               # 入力サイズ（長辺基準）
                conf=self.conf_threshold,      # 信頼度閾値
                iou=self.iou_threshold,        # IoU閾値（NMS）
                max_det=self.max_detections,   # 最大検出数
                verbose=verbose
            )

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # 最初のフレーム（通常1つ）をパース
            return self._parse_results(results[0], inference_time)

        except Exception as e:
            print(f"推論エラー: {e}")
            return None

    def _parse_results(self, result, inference_time: float) -> Dict:
        """
        YOLOの出力から必要な情報を抽出して辞書化

        Args:
            result: YOLOの各フレームごとの結果オブジェクト
            inference_time: 推論時間（秒）

        Returns:
            dict: 検出情報と統計情報を含む辞書
        """
        detections = []
        if result.boxes is not None:
            # 各検出ボックスの情報を取得
            boxes = result.boxes.xyxy.cpu().numpy()        # 座標 (x1, y1, x2, y2)
            confidences = result.boxes.conf.cpu().numpy()  # 信頼度
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # クラスID

            for i in range(len(boxes)):
                class_id = class_ids[i]
                detection = {
                    "bbox": boxes[i].tolist(),
                    "confidence": float(confidences[i]),
                    "class_id": class_id,
                    "class_name": self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                }
                detections.append(detection)

        self.total_detections += len(detections)

        return {
            "detections": detections,
            "inference_time": inference_time,
            "num_detections": len(detections),
            "frame_shape": result.orig_shape  # 元画像サイズ
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """
        推論時間とFPSの統計情報を取得

        Returns:
            dict: 平均・最大・最小推論時間、FPSなど
        """
        if not self.inference_times:
            return {}
        times = np.array(self.inference_times)
        return {
            "avg_inference_time": times.mean(),
            "max_inference_time": times.max(),
            "min_inference_time": times.min(),
            "avg_fps": 1.0 / times.mean() if times.mean() > 0 else 0,
            "total_inferences": len(times),
            "total_detections": self.total_detections
        }

    def reset_stats(self):
        """
        推論履歴・統計をリセット（再計測したいときに使用）
        """
        self.inference_times.clear()
        self.total_detections = 0

    def get_model_info(self) -> Dict[str, any]:
        """
        現在のモデル設定情報を取得

        Returns:
            dict: モデル名・デバイス・閾値・クラス数などの情報
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "num_classes": len(self.class_names),
        }

"""
Detection utilities for YOLO real-time object detection
物体検出結果の描画とユーティリティ機能を提供
"""

import cv2
import numpy as np
import colorsys
import random
from typing import List, Tuple, Dict, Optional
from config import (
    get_class_names,
    VISUALIZATION_CONFIG,
    COLOR_CONFIG,
    FILTER_CONFIG,
)

class DetectionVisualizer:
    """物体検出結果の可視化を担当するクラス"""

    def __init__(self, language: str = "en"):
        """
        Args:
            language: 表示言語（"en" または "jp"）
        """
        self.class_names = get_class_names(language)
        self.colors = self._generate_colors(len(self.class_names))

        # 表示設定の取得（config.pyから）
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = VISUALIZATION_CONFIG["font_scale"]
        self.font_thickness = VISUALIZATION_CONFIG["font_thickness"]
        self.label_padding = VISUALIZATION_CONFIG["label_padding"]
        self.bbox_thickness = VISUALIZATION_CONFIG["bbox_thickness"]

        self.show_confidence = VISUALIZATION_CONFIG["show_confidence"]
        self.show_class_name = VISUALIZATION_CONFIG["show_class_name"]
        self.show_fps = VISUALIZATION_CONFIG["show_fps"]
        self.show_detection_count = VISUALIZATION_CONFIG["show_detection_count"]
        self.show_model_name = VISUALIZATION_CONFIG["show_model_name"]

        self.text_color = COLOR_CONFIG["text_color"]
        self.info_bg_color = COLOR_CONFIG["info_bg_color"]

    def _get_scaling_factor(self, image_shape) -> float:
        """
        入力画像の高さに基づいてスケール係数を返す（720pを基準にスケーリング）
        """
        height = image_shape[0]
        return height / 720

    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """
        クラスごとの色を生成（ランダム or 固定）
        """
        colors = []
        if COLOR_CONFIG["use_random_colors"]:
            # HSV空間を使ってランダムに色を生成
            sat = COLOR_CONFIG["color_saturation"]
            val = COLOR_CONFIG["color_value"]
            for i in range(num_classes):
                hue = i / num_classes
                s = max(0.0, min(1.0, sat + random.uniform(-0.1, 0.1)))
                v = max(0.0, min(1.0, val + random.uniform(-0.1, 0.1)))
                rgb = colorsys.hsv_to_rgb(hue, s, v)
                bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
                colors.append(bgr)
        else:
            # 固定色パレットを繰り返し利用
            base_colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (128, 0, 0), (0, 128, 0), (0, 0, 128)
            ]
            colors = (base_colors * ((num_classes // len(base_colors)) + 1))[:num_classes]
        return colors

    def draw_bounding_box(self, image, bbox, class_id, confidence):
        """
        バウンディングボックスとラベルを画像に描画
        """
        x1, y1, x2, y2 = map(int, bbox)
        color = self.colors[class_id % len(self.colors)]

        # 解像度に応じたスケーリング
        scale = self._get_scaling_factor(image.shape)
        font_scale = self.font_scale * scale
        font_thickness = max(1, int(self.font_thickness * scale))
        bbox_thickness = max(1, int(self.bbox_thickness * scale))
        padding = int(self.label_padding * scale)

        # ボックスを描画
        cv2.rectangle(image, (x1, y1), (x2, y2), color, bbox_thickness)

        # ラベル文字列を構築
        label_parts = []
        if self.show_class_name:
            name = self.class_names[class_id] if class_id < len(self.class_names) else f"ID{class_id}"
            label_parts.append(name)
        if self.show_confidence:
            label_parts.append(f"{confidence:.2f}")
        label = " : ".join(label_parts)

        # ラベル背景と文字を描画
        (w, h), _ = cv2.getTextSize(label, self.font, font_scale, font_thickness)
        cv2.rectangle(image, (x1, y1 - h - 2 * padding), (x1 + w + 2 * padding, y1), color, -1)
        cv2.putText(image, label, (x1 + padding, y1 - padding),
                    self.font, font_scale, self.text_color, font_thickness)
        return image

    def draw_detections(self, image, detections):
        """
        検出結果をすべて描画
        """
        for det in detections:
            image = self.draw_bounding_box(image, det['bbox'], det['class_id'], det['confidence'])
        return image

    def add_info_overlay(self, image, fps=None, detection_count=None, model_name=None, processing_time=None):
        """
        フレームの上に各種統計情報をオーバーレイ表示
        """
        result_image = image.copy()
        info = []

        if fps is not None and self.show_fps:
            info.append(f"FPS: {fps:.1f}")
        if detection_count is not None and self.show_detection_count:
            info.append(f"Detections: {detection_count}")
        if model_name is not None and self.show_model_name:
            info.append(f"Model: {model_name}")
        if processing_time is not None:
            info.append(f"Proc: {processing_time:.1f}ms")

        # スケーリング設定
        scale = self._get_scaling_factor(image.shape)
        font_scale = self.font_scale * scale
        font_thickness = max(1, int(self.font_thickness * scale))
        padding = int(10 * scale)
        line_spacing = int(40 * scale)
        y0 = int(padding + line_spacing)

        # 情報を1行ずつ描画
        for i, text in enumerate(info):
            y = y0 + i * line_spacing
            (w, h), _ = cv2.getTextSize(text, self.font, font_scale, font_thickness)
            cv2.rectangle(result_image, (padding, y - h - padding), (padding + w + 10, y + padding),
                          self.info_bg_color, -1)
            cv2.putText(result_image, text, (padding + 5, y), self.font,
                        font_scale, self.text_color, font_thickness)

        return result_image

    def apply_mosaic(self, image: np.ndarray, detections: List[Dict], class_name: str = "person", mosaic_scale: float = 0.05) -> np.ndarray:
        """
        指定クラス（例: person）にモザイク処理を適用
        """
        result_image = image.copy()
        for det in detections:
            if det["class_name"] != class_name:
                continue

            x1, y1, x2, y2 = map(int, det["bbox"])
            roi = result_image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # モザイク処理（縮小→拡大）
            small = cv2.resize(roi, (0, 0), fx=mosaic_scale, fy=mosaic_scale, interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            result_image[y1:y2, x1:x2] = mosaic

        return result_image

class DetectionFilter:
    """検出結果のフィルタリング機能を提供するクラス"""

    def __init__(self):
        # 設定ファイルからフィルタ条件を取得
        self.conf_thresh = FILTER_CONFIG["confidence_threshold"]
        self.min_area = FILTER_CONFIG["min_bbox_area"]
        self.max_area = FILTER_CONFIG["max_bbox_area"]
        self.allowed_classes = FILTER_CONFIG["allowed_classes"]

    def filter_by_confidence(self, detections: List[Dict], threshold=None):
        """
        信頼度スコアによるフィルタリング
        """
        threshold = threshold if threshold is not None else self.conf_thresh
        return [d for d in detections if d["confidence"] >= threshold]

    def filter_by_area(self, detections: List[Dict], min_area=None, max_area=None):
        """
        バウンディングボックスの面積に基づくフィルタリング
        """
        min_area = min_area if min_area is not None else self.min_area
        max_area = max_area if max_area is not None else self.max_area
        filtered = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area and (max_area is None or area <= max_area):
                filtered.append(d)
        return filtered

    def filter_by_class(self, detections: List[Dict], allowed_classes=None):
        """
        クラスIDに基づくフィルタリング
        """
        allowed = allowed_classes if allowed_classes is not None else self.allowed_classes
        if allowed is None:
            return detections
        return [d for d in detections if d["class_id"] in allowed]

    def apply_all_filters(self, detections: List[Dict]) -> List[Dict]:
        """
        上記3つのフィルタを順に適用して返す
        """
        result = self.filter_by_confidence(detections)
        result = self.filter_by_area(result)
        result = self.filter_by_class(result)
        return result

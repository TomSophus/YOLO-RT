"""
Camera Handler for YOLO Real-time Object Detection
カメラの初期化、フレーム取得、設定管理を行うモジュール
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any, List
from config import CAMERA_CONFIG


class CameraHandler:
    """
    Webカメラの操作を管理するクラス = cv2.VideoCapture() のラッパー
    
    難しくて扱いづらい低レベルAPI(cv2.VideoCapture)を自作のクラス(CameraHandler)
    で包み込んで、扱いやすくしたのが「ラッパー」
    """

    def __init__(self):
        """
        カメラハンドラーの初期化（config.pyから設定を読み込む）
        """
        self.camera_id = CAMERA_CONFIG["camera_id"]
        self.width = CAMERA_CONFIG["width"]
        self.height = CAMERA_CONFIG["height"]
        self.fps = CAMERA_CONFIG["fps"]
        self.cap = None  # 実体のcv2.VideoCaptureオブジェクト
        self.is_opened = False
        self.frame_count = 0  # FPS計測用フレームカウント
        self.start_time = time.time()

    def initialize_camera(self) -> bool:
        """
        カメラを初期化する

        Returns:
            bool: 初期化成功時True
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                print(f"エラー: カメラID {self.camera_id} を開けません")
                return False

            # カメラの映像サイズとフレームレートを設定
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
            # バッファサイズ設定（遅延を防ぐために小さくする）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_CONFIG.get("buffer_size", 1))

            # 実際に設定された値を確認
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"カメラ初期化成功:")
            print(f"  解像度: {actual_width}x{actual_height} (要求: {self.width}x{self.height})")
            print(f"  FPS: {actual_fps} (要求: {self.fps})")

            self.is_opened = True
            return True

        except Exception as e:
            print(f"カメラ初期化エラー: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        カメラからフレームを1枚取得する

        Returns:
            (成功フラグ, 取得した画像データ)
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            return True, frame
        else:
            print("フレーム読み取りエラー")
            return False, None

    def get_fps_stats(self) -> Dict[str, float]:
        """
        経過時間とフレーム数からFPSを算出する

        Returns:
            dict: FPS統計情報（elapsed_time, frame_count, actual_fps, target_fps）
        """
        elapsed_time = time.time() - self.start_time
        actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
        return {
            'elapsed_time': elapsed_time,
            'frame_count': self.frame_count,
            'actual_fps': actual_fps,
            'target_fps': self.fps
        }

    def reset_stats(self):
        """
        FPS計測用のフレーム数・時間をリセットする
        """
        self.frame_count = 0
        self.start_time = time.time()

    def release(self):
        """
        カメラを安全に解放する（OSの占有を解除）
        """
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("カメラリソースを解放しました")

    def __enter__(self):
        """
        with構文で使用できるようにする（初期化処理）
        """
        if self.initialize_camera():
            return self
        else:
            raise RuntimeError("カメラの初期化に失敗しました")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        with構文での終了処理（リソース解放）
        """
        self.release()


def list_available_cameras(max_check: int = 10) -> List[int]:
    """
    利用可能なカメラIDを列挙する関数

    Args:
        max_check (int): 最大でチェックするカメラ番号の上限

    Returns:
        List[int]: 使用可能なカメラIDのリスト
    """
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def test_camera(duration: int = 5) -> bool:
    """
    config.py の設定に基づいてカメラをテストする

    Args:
        duration (int): テスト実行時間（秒）

    Returns:
        bool: テスト成功時 True
    """
    print(f"カメラ {CAMERA_CONFIG['camera_id']} をテスト中... ({duration}秒間)")
    try:
        with CameraHandler() as camera:
            start_time = time.time()
            while time.time() - start_time < duration:
                ret, frame = camera.read_frame()
                if ret:
                    cv2.imshow('Camera Test', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで中断可能
                        break
                else:
                    print("フレーム読み取りエラー")
                    return False

            stats = camera.get_fps_stats()
            print(f"テスト完了: {stats['actual_fps']:.1f} FPS")
            cv2.destroyAllWindows()
            return True

    except Exception as e:
        print(f"カメラテストエラー: {e}")
        return False


if __name__ == "__main__":
    # このファイルを単独実行したときに、カメラ一覧とテストを行う
    print("利用可能なカメラ:", list_available_cameras())
    test_camera(duration=10)

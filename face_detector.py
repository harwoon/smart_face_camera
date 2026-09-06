"""
얼굴 검출
"""
import cv2

import config


class FaceDetector:
    """Haar Cascade 기반 얼굴 검출기.

    detect(frame) -> [(x, y, w, h), ...]  (numpy 배열)
    """

    def __init__(self, cascade_name=None):
        cascade_name = cascade_name or config.CASCADE_NAME
        path = cv2.data.haarcascades + cascade_name
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"Cascade 로드 실패: {path}")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 조명 편차가 심하면 아래 한 줄이 검출률을 크게 올려준다.
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=config.FACE_SCALE_FACTOR,
            minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=config.FACE_MIN_SIZE,
        )
        return faces

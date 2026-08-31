"""
얼굴 검출.

과제에는 "++하고 싶다면"으로 적혀 있지만 사실상 필수다.
AR 배치, 데이터 수집, 실시간 분류가 전부 이 좌표를 쓴다.
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


# TODO(선택): Haar가 잘 안 잡히면 DNN 검출기로 교체를 고려해볼 것.
#   cv2.dnn.readNetFromCaffe(...) 방식이 정면이 아닌 얼굴에도 훨씬 강하다.
#   교체하더라도 detect()가 [(x, y, w, h), ...]를 반환하기만 하면
#   나머지 코드는 하나도 안 고쳐도 된다. 인터페이스를 지키는 게 중요.

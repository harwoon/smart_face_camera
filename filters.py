"""
영상 필터 (요구사항 2).

[ 규칙 ]
모든 필터 함수의 시그니처는 반드시 다음과 같다.

    def f_xxx(frame: np.ndarray) -> np.ndarray

입력도 출력도 3채널 BGR (H, W, 3) 이어야 한다.

**가장 흔한 버그**
grayscale / Canny / threshold 결과는 2차원 (H, W)이다.
그대로 반환하면 AR 합성 단계에서 broadcast 에러가 난다.
반드시 cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)로 3채널로 되돌려서 반환할 것.
아래 to_bgr() 헬퍼를 쓰면 된다.
"""
import cv2
import numpy as np


def to_bgr(img):
    """2채널(흑백) 결과를 3채널로 되돌린다. 이미 3채널이면 그대로."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# ------------------------------------------------------------------ 필터 구현
def f_original(frame):
    return frame


def f_gray(frame):
    # 흑백
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return to_bgr(gray)


def f_blur(frame):
    # 가우시안 블러
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    return to_bgr(blur)


def f_sharpen(frame):
    # 샤프닝
    # 3x3 커널을 만들어 cv2.filter2D(frame, -1, kernel)
    # 중앙값이 크고 주변이 음수인 커널. 커널 합이 1이 되어야 밝기가 유지된다.
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(frame, -1, kernel)


def f_canny(frame):
    # 엣지 검출
    # BGR -> GRAY -> cv2.Canny(gray, th1, th2)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,30,100)
    #   결과가 2차원이므로 to_bgr()로 반환
    return to_bgr(canny)


def f_threshold(frame):
    # 이진화
    # cv2.threshold는 (retval, dst) 두 개를 반환
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    threshold, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) 
    #   조명이 고르지 않으면 cv2.adaptiveThreshold가 훨씬 결과가 좋다.
    return dst


def f_hist_eq(frame):
    # 히스토그램 평활화
    # 컬러 이미지에 그냥 못 쓴다. BGR 각 채널에 따로 걸면 색이 망가진다.
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    # BGR -> YCrCb 변환 후 Y(밝기) 채널에만 equalizeHist를 적용하고 다시 BGR로 되돌린다.
    
    return cv2.cvtColor(ycrcb,cv2.COLOR_YCrCb2BGR)


def f_sepia(frame):
    """세피아 톤. 3x3 색 변환 행렬을 픽셀마다 곱한다.
 
    표준 세피아 공식(RGB 기준)을 OpenCV의 BGR 채널 순서에 맞게
    행과 열을 뒤집어 넣었다.
      R' = 0.393R + 0.769G + 0.189B
      G' = 0.349R + 0.686G + 0.168B
      B' = 0.272R + 0.534G + 0.131B
    """
    kernel = np.array([[0.272, 0.534, 0.131],   # B
                        [0.349, 0.686, 0.168],   # G
                        [0.393, 0.769, 0.189]])  # R
 
    sepia = cv2.transform(frame, kernel)
    # cv2.transform은 255를 넘는 값을 자동으로 clip하지 않고 wraparound시킨다.
    # (예: 260 -> 4) clip을 빼면 밝은 영역에 검은 반점이 생기는 버그가 난다.
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


# ------------------------------------------------------------------ 레지스트리
# 숫자키와 필터 연결
FILTERS = {
    0: ("ORIGINAL", f_original),
    1: ("GRAY", f_gray),
    2: ("BLUR", f_blur),
    3: ("SHARPEN", f_sharpen),
    4: ("CANNY", f_canny),
    5: ("THRESHOLD", f_threshold),
    6: ("HIST_EQ", f_hist_eq),
    7: ("SEPIA", f_sepia),
}


def apply_filter(frame, filter_id):
    """filter_id에 해당하는 필터를 적용. 없는 id면 원본 그대로."""
    entry = FILTERS.get(filter_id)
    if entry is None:
        return frame
    _, func = entry
    return to_bgr(func(frame))


def filter_name(filter_id):
    entry = FILTERS.get(filter_id)
    return entry[0] if entry else "UNKNOWN"

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
    """구현 예시. 이 패턴을 그대로 따라 나머지를 채우면 된다."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return to_bgr(gray)


def f_blur(frame):
    # TODO: Gaussian Blur 구현
    #   힌트) cv2.GaussianBlur(frame, (커널크기, 커널크기), 시그마)
    #   커널 크기는 반드시 홀수여야 한다. 짝수면 에러.
    return frame


def f_sharpen(frame):
    # TODO: 샤프닝 구현
    #   힌트) 3x3 커널을 만들어 cv2.filter2D(frame, -1, kernel)
    #   중앙값이 크고 주변이 음수인 커널. 커널 합이 1이 되어야 밝기가 유지된다.
    return frame


def f_canny(frame):
    # TODO: 엣지 검출 구현
    #   힌트) BGR -> GRAY -> cv2.Canny(gray, th1, th2)
    #   결과가 2차원이므로 to_bgr()로 감싸서 반환할 것!
    return frame


def f_threshold(frame):
    # TODO: 이진화 구현
    #   힌트) cv2.threshold는 (retval, dst) 두 개를 반환한다. 두 번째만 쓴다.
    #   조명이 고르지 않으면 cv2.adaptiveThreshold가 훨씬 결과가 좋다.
    return frame


def f_hist_eq(frame):
    # TODO: 히스토그램 평활화 구현
    #   주의) 컬러 이미지에 그냥 못 쓴다. BGR 각 채널에 따로 걸면 색이 망가진다.
    #   힌트) BGR -> YCrCb 변환 후 Y(밝기) 채널에만 equalizeHist를 적용하고
    #         다시 BGR로 되돌린다.
    return frame


def f_cartoon(frame):
    # TODO(선택): 직접 구성한 효과. 가산점 노리는 자리.
    #   예) bilateralFilter로 색을 뭉갠 뒤 adaptiveThreshold 윤곽선을 곱하기
    #   예) 세피아 톤 (3x3 색 변환 행렬 + cv2.transform)
    return frame


# ------------------------------------------------------------------ 레지스트리
# 숫자키와 필터를 연결한다. 새 필터를 추가하려면 여기 한 줄만 추가하면 된다.
FILTERS = {
    0: ("ORIGINAL", f_original),
    1: ("GRAY", f_gray),
    2: ("BLUR", f_blur),
    3: ("SHARPEN", f_sharpen),
    4: ("CANNY", f_canny),
    5: ("THRESHOLD", f_threshold),
    6: ("HIST_EQ", f_hist_eq),
    7: ("CARTOON", f_cartoon),
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

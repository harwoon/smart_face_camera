"""
공용 유틸리티.

여기 있는 함수는 "직접 짜면 반드시 버그가 나는 것들"만 모아뒀다.
- clamp_box  : 얼굴이 화면 밖으로 나갈 때 프로그램이 죽는 문제 방지
- crop_face  : 학습/추론이 완전히 동일한 방식으로 얼굴을 자르도록 통일
- imread/imwrite_unicode : Windows 한글 경로 문제 우회
"""
import cv2
import numpy as np

import config


# ------------------------------------------------------------------ 박스 처리
def clamp_box(x, y, w, h, frame_w, frame_h):
    """박스를 프레임 안쪽으로 자른다. 잘린 뒤 크기가 0이면 None 반환.

    이걸 안 하면 얼굴을 화면 가장자리로 옮기는 순간
    ROI 크기가 안 맞아서 프로그램이 죽는다. (시연 중 가장 흔한 사고)
    """
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_w, int(x + w))
    y2 = min(frame_h, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def expand_box(x, y, w, h, margin, frame_w, frame_h):
    """박스를 margin 비율만큼 키운 뒤 프레임 안으로 clamp."""
    dx = int(w * margin)
    dy = int(h * margin)
    return clamp_box(x - dx, y - dy, w + 2 * dx, h + 2 * dy, frame_w, frame_h)


# ------------------------------------------------------------------ 얼굴 크롭
def crop_face(frame, face, margin=None, size=None):
    """얼굴 영역을 잘라 정사각형으로 리사이즈해 반환.

    데이터 수집(collector.py)과 실시간 추론(recognizer.py)이
    반드시 이 함수 하나만 사용해야 한다. 각자 따로 짜면 어긋난다.
    """
    margin = config.FACE_MARGIN if margin is None else margin
    size = config.IMG_SIZE if size is None else size

    h_frame, w_frame = frame.shape[:2]
    box = expand_box(*face, margin, w_frame, h_frame)
    if box is None:
        return None
    x, y, w, h = box
    return cv2.resize(frame[y:y + h, x:x + w], (size, size))


def largest_face(faces):
    """검출된 얼굴 중 가장 큰 것 하나. 없으면 None."""
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


# ------------------------------------------------------------------ 파일 입출력
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """한글 경로 대응 imread. cv2.imread는 Windows 한글 경로에서 None을 반환한다."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path, img):
    """한글 경로 대응 imwrite."""
    path = str(path)
    ext = path[path.rfind('.'):]
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ 화면 표시
def draw_text(frame, text, org, color=(255, 255, 255), scale=0.6, thickness=2):
    """외곽선을 넣어 배경과 상관없이 읽히는 텍스트.

    주의: cv2.putText는 한글을 못 그린다. 영어만 사용할 것.
    (한글이 꼭 필요하면 PIL ImageDraw + 폰트를 써야 한다)
    """
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)
    return frame


def draw_face_boxes(frame, faces, color=(0, 255, 0)):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame

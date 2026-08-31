"""
AR 아이템 합성 (요구사항 3).

[ 규칙 ]
모든 아이템 함수의 시그니처는 다음과 같다.

    def item_xxx(frame, face) -> frame
    # face = (x, y, w, h)  얼굴 박스 하나

overlay()는 이미 완성되어 있으므로, 각 아이템 함수에서는
"얼굴 박스 기준으로 어디에 얼마 크기로 놓을지" 좌표만 계산하면 된다.

[ PNG 준비 ]
assets/ 폴더에 배경이 투명한 PNG를 넣을 것.
반드시 알파 채널(4채널)이 있어야 한다. 없으면 검은 사각형이 붙는다.
"""
import cv2
import numpy as np

import config
from utils import clamp_box

_cache = {}


def load_item(filename):
    """assets/에서 PNG를 알파 채널 포함해 읽는다. 결과는 캐싱."""
    if filename in _cache:
        return _cache[filename]

    path = config.ASSETS_DIR / filename
    # IMREAD_UNCHANGED가 핵심. 기본값(IMREAD_COLOR)으로 읽으면
    # 알파 채널이 날아가서 투명 처리가 안 된다.
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"[AR] 파일 없음: {path}")
    elif img.shape[2] != 4:
        print(f"[AR] 경고: 알파 채널 없음 ({filename}). 배경이 그대로 붙습니다.")

    _cache[filename] = img
    return img


def overlay(frame, item, x, y, w, h):
    """알파 블렌딩으로 item을 frame 위에 합성. (완성본 - 수정 불필요)

    화면 밖으로 나가는 부분은 자동으로 잘라내므로 얼굴이 가장자리로 가도
    프로그램이 죽지 않는다.
    """
    if item is None or w <= 0 or h <= 0:
        return frame

    fh, fw = frame.shape[:2]

    # 1) 그릴 영역을 프레임 안으로 clamp
    box = clamp_box(x, y, w, h, fw, fh)
    if box is None:
        return frame
    cx, cy, cw, ch = box

    # 2) 아이템을 원래 목표 크기로 리사이즈한 뒤,
    #    잘려나간 만큼 아이템 쪽도 똑같이 잘라낸다.
    resized = cv2.resize(item, (int(w), int(h)))
    ox = cx - int(x)          # 왼쪽/위쪽이 잘린 픽셀 수
    oy = cy - int(y)
    resized = resized[oy:oy + ch, ox:ox + cw]

    roi = frame[cy:cy + ch, cx:cx + cw]

    if resized.shape[2] == 4:
        alpha = resized[:, :, 3:4].astype(np.float32) / 255.0   # (h, w, 1)
        fg = resized[:, :, :3].astype(np.float32)
        blended = roi.astype(np.float32) * (1 - alpha) + fg * alpha
        frame[cy:cy + ch, cx:cx + cw] = blended.astype(np.uint8)
    else:
        frame[cy:cy + ch, cx:cx + cw] = resized[:, :, :3]

    return frame


# ------------------------------------------------------------------ 아이템 구현
def item_hat(frame, face):
    """구현 예시 겸 참고용. 비율은 실제 PNG에 맞게 조정할 것."""
    x, y, w, h = face
    item = load_item("hat.png")

    item_w = int(w * 1.2)                    # 얼굴보다 약간 넓게
    item_h = int(item_w * 0.7)               # PNG의 가로세로 비율에 맞춰 조정
    item_x = x - (item_w - w) // 2           # 가로 중앙 정렬
    item_y = y - int(item_h * 0.8)           # 이마 위쪽으로 올림

    return overlay(frame, item, item_x, item_y, item_w, item_h)


def item_glasses(frame, face):
    # TODO: 선글라스 배치
    #   힌트) 눈은 대략 얼굴 박스 위에서 25~35% 지점에 있다.
    #         item_y = y + int(h * 0.25) 부근부터 시작해서 눈으로 맞춰볼 것.
    #   폭은 얼굴 폭(w)과 비슷하거나 살짝 크게.
    return frame


def item_mustache(frame, face):
    # TODO: 수염/마스크 배치
    #   힌트) 코 아래는 대략 얼굴 박스 위에서 60~65% 지점.
    #         폭은 얼굴 폭의 절반 정도가 자연스럽다.
    return frame


def item_extra(frame, face):
    # TODO(선택): 네 번째 아이템. 자유롭게.
    #   아이디어) 얼굴 위에 이름표, 하트 이펙트, 강아지 귀+코 세트 등
    #   여러 PNG를 한 함수에서 여러 번 overlay 해도 된다.
    return frame


# ------------------------------------------------------------------ 레지스트리
AR_ITEMS = {
    0: ("NONE", None),
    1: ("HAT", item_hat),
    2: ("GLASSES", item_glasses),
    3: ("MUSTACHE", item_mustache),
    4: ("EXTRA", item_extra),
}


def apply_ar(frame, ar_id, faces):
    """검출된 모든 얼굴에 아이템을 적용."""
    entry = AR_ITEMS.get(ar_id)
    if entry is None or entry[1] is None:
        return frame

    _, func = entry
    for face in faces:
        frame = func(frame, face)
    return frame


def ar_name(ar_id):
    entry = AR_ITEMS.get(ar_id)
    return entry[0] if entry else "NONE"

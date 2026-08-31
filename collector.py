"""
얼굴 데이터 수집 (요구사항 4).

이 단계의 데이터 품질이 5, 6번의 성패를 그대로 결정한다.
모델을 튜닝하는 것보다 여기서 좋은 데이터를 모으는 게 훨씬 중요하다.
"""
import time

import cv2

import config
from utils import crop_face, draw_text, imwrite_unicode, largest_face


class FaceCollector:
    """상태를 들고 있다가 main 루프에서 매 프레임 update()를 호출받는다.

    사용법:
        collector.start("wonhee")
        frame, done = collector.update(frame, faces)
    """

    def __init__(self, target=None, interval=None):
        self.target = target or config.COLLECT_TARGET
        self.interval = interval or config.COLLECT_INTERVAL
        self.reset()

    def reset(self):
        self.name = None
        self.save_dir = None
        self.count = 0
        self.frame_idx = 0
        self.started_at = None

    def start(self, name):
        """이름을 받아 폴더를 만들고 수집을 시작한다."""
        name = name.strip()
        if not name:
            return False

        # 경로에 한글을 쓰면 OS/라이브러리마다 문제가 생기므로 영문 권장.
        # ImageFolder의 클래스명이 그대로 이 폴더명이 된다.
        self.name = name
        self.save_dir = config.DATASET_DIR / name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 이어서 수집할 수 있도록 기존 파일 개수부터 시작
        self.count = len(list(self.save_dir.glob("*.jpg")))
        self.frame_idx = 0
        self.started_at = time.time()
        print(f"[수집] '{name}' 시작 (기존 {self.count}장) -> {self.save_dir}")
        return True

    def update(self, raw_frame, faces):
        """한 프레임 저장 처리. 완료 여부(bool)를 반환한다.

        주의: raw_frame은 '필터/AR이 적용되기 전의 원본'이어야 한다.
        필터가 걸린 화면을 저장하면 학습 데이터가 오염된다.
        화면 표시는 draw_hud()가 따로 담당한다.
        """
        if self.name is None:
            return True

        self.frame_idx += 1
        face = largest_face(faces)

        if face is not None and self.frame_idx % self.interval == 0:
            # 매 프레임 저장하면 초당 30장의 '거의 똑같은 사진'이 쌓인다.
            # 간격을 둬서 그 사이에 자세가 바뀌도록 유도한다.
            crop = crop_face(raw_frame, face)
            if crop is not None:
                filename = self.save_dir / f"{self.name}_{self.count:04d}.jpg"
                if imwrite_unicode(filename, crop):
                    self.count += 1

        if self.count >= self.target:
            print(f"[수집] '{self.name}' 완료: {self.count}장")
            self.reset()
            return True

        return False

    def draw_hud(self, frame, faces=()):
        """표시용 프레임에 진행 상황을 그린다."""
        if self.name is None:
            return frame

        if len(faces) == 0:
            draw_text(frame, "NO FACE DETECTED", (20, 120), (0, 0, 255))

        draw_text(frame, f"REGISTERING: {self.name}", (20, 30), (0, 255, 255))
        draw_text(frame, f"{self.count} / {self.target}", (20, 60), (0, 255, 255))

        # 진행 바
        h, w = frame.shape[:2]
        ratio = min(1.0, self.count / self.target)
        cv2.rectangle(frame, (20, h - 40), (w - 20, h - 20), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 40),
                      (20 + int((w - 40) * ratio), h - 20), (0, 200, 255), -1)

        # 다양성 확보를 위한 안내. 이게 정확도를 크게 좌우한다.
        tips = [
            "Turn head LEFT / RIGHT",
            "Look UP / DOWN",
            "Change expression",
            "Move closer / farther",
        ]
        tip = tips[int(self.count / max(1, self.target / len(tips))) % len(tips)]
        draw_text(frame, tip, (20, 90), (255, 255, 255), scale=0.55)
        return frame


# TODO(선택): 수집 품질을 더 올리고 싶다면
#   1) 블러 판정 - cv2.Laplacian(gray, cv2.CV_64F).var() 값이 낮으면
#      흔들린 사진이므로 저장하지 않고 건너뛴다.
#   2) 조명 다양화 - 창가/실내등 등 조명을 바꿔가며 두세 번 나눠 수집한다.
#   3) 배경 다양화 - 같은 자리에서만 찍으면 모델이 배경을 외워버린다.

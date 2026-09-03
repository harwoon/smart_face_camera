"""
실시간 얼굴 분류 (요구사항 6).

[ 가장 많이 터지는 버그: BGR / RGB ]
OpenCV는 BGR로 읽는다. 반면 ImageFolder + PIL로 학습했다면 모델은 RGB를
기대한다. 그대로 넣으면 채널이 뒤집힌 이미지를 넣는 셈인데,
에러는 안 나고 정확도만 이상하게 떨어져서 원인 찾기가 정말 어렵다.
반드시 cv2.cvtColor(..., cv2.COLOR_BGR2RGB)를 거칠 것.
"""
from collections import Counter, deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from train import build_model
from utils import crop_face, get_device


class Recognizer:
    """모델 로드 + 얼굴 크롭 추론 + 결과 스무딩."""

    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or config.MODEL_PATH
        self.device = device or get_device()

        self.model = None
        self.classes = []
        self.frame_idx = 0
        self.history = deque(maxlen=config.VOTE_WINDOW)
        self.last_result = None      # (label, confidence)

        # 학습의 val_transform과 완전히 동일해야 한다.
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])

    # -------------------------------------------------------------- 로드
    def load(self):
        if not self.model_path.exists():
            print(f"[인식] 모델 없음: {self.model_path}. 먼저 train.py를 실행하세요.")
            return False

        ckpt = torch.load(self.model_path, map_location=self.device)
        self.classes = ckpt["classes"]          # 학습 때 저장된 순서 그대로 사용

        self.model = build_model(len(self.classes), backbone=ckpt.get("arch", "resnet18"))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()                       # 필수. BatchNorm/Dropout 동작이 달라진다.

        print(f"[인식] 모델 로드 완료. 클래스: {self.classes}")
        return True

    @property
    def ready(self):
        return self.model is not None

    # -------------------------------------------------------------- 추론
    def predict(self, frame, face):
        """얼굴 하나를 분류. (label, confidence) 반환.

        frame은 필터/AR이 적용되기 전의 원본이어야 한다.
        """
        crop = crop_face(frame, face)           # 학습 때와 동일한 크롭 방식
        if crop is None:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)   # BGR -> RGB (중요!)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():                   # 필수. 없으면 느리고 메모리도 샌다.
            probs = F.softmax(self.model(tensor), dim=1)[0]

        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])

    def update(self, frame, faces):
        """매 프레임 호출. INFER_INTERVAL마다 한 번만 실제 추론한다.

        매 프레임 추론하면 눈에 띄게 렉이 걸린다.
        """
        self.frame_idx += 1

        if len(faces) == 0:
            self.history.clear()
            self.last_result = None
            return None

        if self.frame_idx % config.INFER_INTERVAL == 0:
            from utils import largest_face
            result = self.predict(frame, largest_face(faces))
            if result:
                self.history.append(result[0])
                # 최근 N회의 최빈값을 쓰면 라벨 깜빡임이 사라진다.
                label = Counter(self.history).most_common(1)[0][0]
                self.last_result = (label, result[1])

        return self.last_result


def draw_result(frame, faces, result):
    """검출 박스 위에 이름과 확률을 표시."""
    from utils import draw_text, largest_face

    face = largest_face(faces)
    if face is None:
        return frame

    x, y, w, h = face
    color = (0, 255, 0) if result else (128, 128, 128)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if result:
        label, conf = result
        text = f"{label} {conf * 100:.1f}%"
        draw_text(frame, text, (x, max(20, y - 10)), color)

    return frame
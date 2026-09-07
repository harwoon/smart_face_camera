"""
ArcFace 임베딩 기반 얼굴 인식 (실험적 대체 구현)

recognizer.py(CNN 분류기)와 인터페이스가 동일하다(ready/load/update).
main.py는 config.USE_ARCFACE 값으로 둘 중 하나를 골라 쓴다.

분류기를 새로 학습하는 대신, insightface의 사전학습 ArcFace 모델로
얼굴을 512차원 벡터로 바꾼 뒤 등록자별 대표 벡터와 코사인 유사도를 비교한다.
등록 안 된 사람은 모든 등록자와의 유사도가 낮게 나오므로,
"5명 중 누가 제일 비슷한가"를 억지로 고르는 softmax 방식보다 unknown 판별이 잘 된다.

사용 전에 enroll_arcface.py로 dataset/의 사진에서 갤러리(arcface_gallery.npz)를 만들어야 한다.
"""
from collections import Counter, deque

import numpy as np

import config


class ArcFaceRecognizer:
    """insightface 임베딩 + 코사인 유사도 매칭. Recognizer와 동일한 인터페이스."""

    def __init__(self, gallery_path=None):
        self.gallery_path = gallery_path or config.ARCFACE_GALLERY_PATH
        self.app = None
        self.names = []          # 갤러리 순서
        self.prototypes = None   # (N, 512) L2 정규화된 등록자별 대표 임베딩

        self.frame_idx = 0
        self.history = deque(maxlen=config.VOTE_WINDOW)
        self.last_result = None

    # -------------------------------------------------------------- 로드
    def load(self):
        if not self.gallery_path.exists():
            print(f"[ArcFace] 갤러리 없음: {self.gallery_path}. 먼저 enroll_arcface.py를 실행하세요.")
            return False

        from insightface.app import FaceAnalysis
        # allowed_modules로 검출+임베딩만 로드한다. 기본값은 3D/2D 랜드마크,
        # 나이/성별 모델까지 얼굴마다 돌려서 프레임당 733ms -> 122ms로 6배 느렸다.
        self.app = FaceAnalysis(name=config.ARCFACE_MODEL_NAME,
                                 providers=["CPUExecutionProvider"],
                                 allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        data = np.load(self.gallery_path, allow_pickle=True)
        self.names = list(data["names"])
        self.prototypes = data["embeddings"]

        print(f"[ArcFace] 갤러리 로드 완료: {self.names}")
        return True

    @property
    def ready(self):
        return self.app is not None

    # -------------------------------------------------------------- 추론
    def predict(self, frame):
        """프레임 전체에서 얼굴을 검출+임베딩. (label, 유사도) 반환.

        Haar 박스가 아니라 insightface 자체 검출(+정렬)을 쓴다.
        ArcFace는 5점 랜드마크로 정렬된 얼굴이어야 임베딩 품질이 나온다.
        """
        faces = self.app.get(frame)
        if not faces:
            return None

        target = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = target.normed_embedding  # 이미 L2 정규화됨

        sims = self.prototypes @ emb
        idx = int(np.argmax(sims))
        return self.names[idx], float(sims[idx])

    def update(self, frame, faces):
        """매 프레임 호출. INFER_INTERVAL마다 한 번만 실제 추론한다."""
        self.frame_idx += 1

        if len(faces) == 0:
            self.history.clear()
            self.last_result = None
            return None

        if self.frame_idx % config.INFER_INTERVAL == 0:
            result = self.predict(frame)
            if result is None:
                self.last_result = None
                return self.last_result

            label, sim = result
            if sim < config.ARCFACE_SIM_THRESHOLD:
                label = "UNKNOWN"

            # UNKNOWN 판정까지 포함해 다수결을 내야 라벨/유사도가 항상 같은
            # 프레임 것으로 일치하고, 한 프레임만 튀어도 묻히지 않는다.
            self.history.append((label, sim))
            voted_label = Counter(l for l, _ in self.history).most_common(1)[0][0]
            voted_conf = next(c for l, c in reversed(self.history) if l == voted_label)
            self.last_result = (voted_label, voted_conf)

        return self.last_result

"""
프로젝트 전역 설정.

여기 있는 값은 팀원 전체가 공유한다.
특히 FACE_MARGIN / IMG_SIZE / IMAGENET_* 는 학습과 추론이 반드시
같은 값을 써야 한다. 한쪽만 바꾸면 에러 없이 정확도만 무너진다.
"""
from pathlib import Path

# ---------------------------------------------------------------- 경로
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = BASE_DIR / "face_model.pth"

# ---------------------------------------------------------------- 웹캠
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR = True          # 좌우 반전(거울 모드). UX가 훨씬 자연스러워진다.

# ---------------------------------------------------------------- 얼굴 검출
CASCADE_NAME = "haarcascade_frontalface_default.xml"
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (80, 80)

# ---------------------------------------------------------------- 학습/추론 공용
# 얼굴 박스 바깥으로 얼마나 여유를 두고 자를지 (0.2 = 상하좌우 20%)
FACE_MARGIN = 0.2
IMG_SIZE = 224

# ImageNet 사전학습 가중치에 맞춘 정규화 값. 바꾸지 말 것.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------- 데이터 수집
COLLECT_TARGET = 200     # 1인당 목표 장수
COLLECT_INTERVAL = 5     # N프레임마다 1장 저장 (연속 중복 방지)

# ---------------------------------------------------------------- 실시간 추론
INFER_INTERVAL = 5       # N프레임마다 1회만 추론 (매 프레임 하면 렉)
VOTE_WINDOW = 5          # 최근 N회 예측의 최빈값 사용 (라벨 깜빡임 방지)
CONF_THRESHOLD = 0.0     # 과제상 미등록자 판별은 불필요하므로 기본 0

# ---------------------------------------------------------------- 학습 파라미터
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2

# ---------------------------------------------------------------- 키 바인딩
KEY_QUIT = ord('q')
KEY_REGISTER = ord('r')     # 얼굴 등록 모드
KEY_RECOGNIZE = ord('f')    # 얼굴 인식 모드
KEY_NORMAL = ord('n')       # 일반 모드로 복귀
KEY_RELOAD = ord('l')       # 모델 다시 로드
KEY_HELP = ord('h')

# 필터: 숫자키 0~9  /  AR 아이템: a s d f ... 는 f가 겹치므로 z x c 사용
AR_KEYS = {ord('z'): 1, ord('x'): 2, ord('c'): 3, ord('v'): 4}
AR_KEY_OFF = ord('`')

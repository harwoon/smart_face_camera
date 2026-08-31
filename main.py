"""
스마트 얼굴 카메라 - 메인 루프 (요구사항 1)

실행:  python main.py

[ 조작 ]
  0~7  : 필터 선택
  ` z x c v : AR 아이템 (` = 끄기)
  r    : 얼굴 등록 (터미널에 이름 입력)
  f    : 얼굴 인식 모드
  n    : 일반 모드
  l    : 모델 다시 로드
  h    : 도움말 표시 토글
  q    : 종료

[ 설계 원칙 ]
필터 / AR / 모드는 서로 독립된 상태 변수다.
하나로 뭉뚱그리면 "그레이스케일 + 모자" 같은 조합이 안 나온다.
"""
import cv2

import ar_items
import config
import filters
from collector import FaceCollector
from face_detector import FaceDetector
from utils import draw_face_boxes, draw_text

MODE_NORMAL = "NORMAL"
MODE_REGISTER = "REGISTER"
MODE_RECOGNIZE = "RECOGNIZE"


class SmartCamera:
    def __init__(self):
        self.cap = None
        self.detector = FaceDetector()
        self.collector = FaceCollector()
        self.recognizer = None          # torch import가 무거우므로 지연 로드

        self.mode = MODE_NORMAL
        self.filter_id = 0
        self.ar_id = 0
        self.show_boxes = True
        self.show_help = True

    # -------------------------------------------------------------- 웹캠
    def open_camera(self):
        self.cap = cv2.VideoCapture(config.CAM_INDEX)
        # Windows에서 열리는 데 오래 걸리면 아래처럼 백엔드를 지정해볼 것
        # self.cap = cv2.VideoCapture(config.CAM_INDEX, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"웹캠을 열 수 없습니다 (index={config.CAM_INDEX})")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        print("[카메라] 시작")

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[카메라] 종료")

    # -------------------------------------------------------------- 모드
    def enter_register(self):
        # cv2 창에서는 텍스트 입력이 안 되므로 터미널로 받는다.
        # 이 동안 영상은 멈추지만 과제 요구사항에는 문제없다.
        name = input("등록할 이름을 입력하세요 (영문 권장): ")
        if self.collector.start(name):
            self.mode = MODE_REGISTER
        else:
            print("[등록] 이름이 비어 있어 취소되었습니다.")

    def enter_recognize(self):
        if self.recognizer is None:
            print("[인식] 모델 로딩 중...")
            from recognizer import Recognizer     # 여기서 지연 import
            self.recognizer = Recognizer()

        if self.recognizer.ready or self.recognizer.load():
            self.mode = MODE_RECOGNIZE
        else:
            print("[인식] 모델이 없어 일반 모드를 유지합니다.")

    # -------------------------------------------------------------- 키 처리
    def handle_key(self, key):
        if key == config.KEY_QUIT:
            return False

        if ord('0') <= key <= ord('9'):
            self.filter_id = key - ord('0')
        elif key in config.AR_KEYS:
            self.ar_id = config.AR_KEYS[key]
        elif key == config.AR_KEY_OFF:
            self.ar_id = 0
        elif key == config.KEY_REGISTER:
            self.enter_register()
        elif key == config.KEY_RECOGNIZE:
            self.enter_recognize()
        elif key == config.KEY_NORMAL:
            self.mode = MODE_NORMAL
            self.collector.reset()
        elif key == config.KEY_RELOAD:
            if self.recognizer:
                self.recognizer.load()
        elif key == config.KEY_HELP:
            self.show_help = not self.show_help
        elif key == ord('b'):
            self.show_boxes = not self.show_boxes

        return True

    # -------------------------------------------------------------- HUD
    def draw_hud(self, frame):
        h = frame.shape[0]
        status = (f"MODE:{self.mode}  "
                  f"FILTER:{filters.filter_name(self.filter_id)}  "
                  f"AR:{ar_items.ar_name(self.ar_id)}")
        draw_text(frame, status, (10, h - 15), (200, 255, 200), scale=0.5)

        if self.show_help:
            lines = ["0-9 filter", "` z x c v  AR", "r register",
                     "f recognize", "n normal", "h help", "q quit"]
            for i, line in enumerate(lines):
                draw_text(frame, line, (frame.shape[1] - 150, 25 + i * 22),
                          (220, 220, 220), scale=0.45, thickness=1)
        return frame

    # -------------------------------------------------------------- 메인 루프
    def run(self):
        self.open_camera()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[카메라] 프레임을 읽지 못했습니다.")
                    break

                if config.MIRROR:
                    frame = cv2.flip(frame, 1)

                # 원본은 따로 보관한다.
                # 데이터 저장과 추론은 반드시 필터/AR이 안 걸린 원본으로 해야 한다.
                raw = frame.copy()

                faces = self.detector.detect(raw)

                # 화면용 처리
                frame = filters.apply_filter(frame, self.filter_id)
                frame = ar_items.apply_ar(frame, self.ar_id, faces)

                if self.mode == MODE_REGISTER:
                    # 저장은 raw에서, HUD는 frame에
                    done = self.collector.update(raw, faces)
                    if done:
                        self.mode = MODE_NORMAL
                    else:
                        frame = draw_face_boxes(frame, faces, (0, 255, 255))
                        frame = self.collector.draw_hud(frame, faces)

                elif self.mode == MODE_RECOGNIZE:
                    from recognizer import draw_result
                    result = self.recognizer.update(raw, faces)
                    frame = draw_result(frame, faces, result)

                elif self.show_boxes:
                    frame = draw_face_boxes(frame, faces)

                frame = self.draw_hud(frame)
                cv2.imshow("Smart Face Camera", frame)

                # & 0xFF 를 빼면 환경에 따라 키 인식이 안 된다.
                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self.handle_key(key):
                    break

                # 창의 X 버튼으로 닫았을 때도 정상 종료되도록
                if cv2.getWindowProperty("Smart Face Camera",
                                         cv2.WND_PROP_VISIBLE) < 1:
                    break

        except KeyboardInterrupt:
            print("\n[종료] 사용자 중단")
        finally:
            self.close_camera()


if __name__ == "__main__":
    SmartCamera().run()

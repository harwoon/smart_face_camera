# 스마트 얼굴 카메라 (OpenCV + CNN 전이학습)

실시간 웹캠 영상에 영상처리 필터와 AR 아이템을 적용하고,
직접 수집한 얼굴 데이터로 전이학습한 CNN 모델을 이용해
등록된 사람을 실시간으로 분류하는 프로그램

## 실행

```bash
pip install -r requirements.txt
python main.py          # 카메라 / 필터 / AR / 등록 / 인식
python train.py         # 수집한 데이터로 전이학습
```

## 조작

| 키 | 동작 |
|---|---|
| `0` ~ `7` | 필터 전환 |
| `` ` `` `z` `x` `c` `v` | AR 아이템 (`` ` ``로 끄기) |
| `r` | 얼굴 등록 (터미널에 이름 입력) |
| `f` | 얼굴 인식 모드 |
| `n` | 일반 모드 |
| `l` | 모델 다시 로드 |
| `b` | 얼굴 박스 표시 토글 |
| `h` | 도움말 토글 |
| `q` | 종료 |

## 파일 구조

```
main.py            메인 루프, 상태 관리, 키 입력 분기
config.py          전역 설정 (여기만 고치면 전체에 반영)
utils.py           박스 clamp, 얼굴 크롭, 한글 경로 IO
face_detector.py   Haar Cascade 얼굴 검출
filters.py         영상 필터            
ar_items.py        AR 아이템 합성       
collector.py       얼굴 데이터 수집      
train.py           전이학습             
recognizer.py      실시간 추론          
assets/            AR용 투명 PNG
dataset/{name}/    수집된 얼굴 이미지 (git 제외)
face_model.pth     학습된 모델 (git 제외)
```

## 인터페이스 규약

```python
detector.detect(frame)              -> [(x, y, w, h), ...]
filters.apply_filter(frame, id)     -> frame   # 입출력 모두 3채널 BGR
ar_items.apply_ar(frame, id, faces) -> frame
collector.update(raw_frame, faces)  -> bool    # 완료 여부
recognizer.update(raw_frame, faces) -> (label, conf) | None
```

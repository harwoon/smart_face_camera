# 스마트 얼굴 카메라 (OpenCV + CNN 전이학습)

실시간 웹캠 영상에 영상처리 필터와 AR 아이템을 적용하고,
직접 수집한 얼굴 데이터로 전이학습한 CNN 모델을 이용해
등록된 사람을 실시간으로 분류하는 프로그램이다.

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

새 필터나 아이템을 추가할 때는 함수를 만들고
`FILTERS` / `AR_ITEMS` 딕셔너리에 한 줄만 등록하면 된다.
`main.py`는 건드릴 일이 없다. (git 충돌 방지)

## 작업 순서

1. **Phase 1** 뼈대 확인 — 웹캠이 뜨고 얼굴에 사각형이 그려지는지
2. **Phase 2** `filters.py`의 TODO 채우기
3. **Phase 3** `assets/`에 PNG 넣고 `ar_items.py`의 TODO 채우기
4. **Phase 4** 팀원 전원 모여서 데이터 수집 (`r` 키)
5. **Phase 5** `python train.py`
6. **Phase 6** `f` 키로 인식 테스트

## 주의사항 (미리 읽으면 몇 시간을 아낀다)

- **필터는 반드시 3채널을 반환할 것.** grayscale/Canny 결과는 2차원이라
  그대로 반환하면 AR 합성에서 터진다. `to_bgr()`로 감쌀 것.
- **AR PNG는 `cv2.IMREAD_UNCHANGED`로 읽을 것.** 기본값으로 읽으면
  알파 채널이 날아가 검은 사각형이 붙는다.
- **경로에 한글을 쓰지 말 것.** `cv2.imread/imwrite`가 Windows 한글 경로에서
  조용히 실패한다. `cv2.putText`도 한글을 못 그린다.
- **저장과 추론은 원본 프레임으로.** 필터 걸린 화면을 저장하면 데이터가 오염된다.
- **`dataset.classes`를 모델과 함께 저장할 것.** 순서가 어긋나면
  이름만 서로 바뀐 채 잘 돌아가는 버그가 난다. (`train.py`에 이미 반영됨)
- **BGR → RGB 변환 필수.** 안 하면 에러 없이 정확도만 떨어진다.
  (`recognizer.py`에 이미 반영됨)
- **수집할 때 고개를 돌리고 표정을 바꿀 것.** 가만히 있으면
  똑같은 사진 200장이라 과적합된다.
- **사람마다 이미지 개수를 비슷하게 맞출 것.**

## 데이터 수집 체크리스트

- [ ] 1인당 150~300장
- [ ] 고개 좌우/상하, 표정 변화, 거리 변화 포함
- [ ] 조명 조건 2가지 이상
- [ ] 배경 2곳 이상 (한 곳에서만 찍으면 배경을 외운다)
- [ ] 인원별 개수 균형
- [ ] 최종 검증용으로 **다른 날/다른 장소** 사진 따로 확보

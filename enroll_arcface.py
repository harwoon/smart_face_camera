"""
dataset/의 사진으로 ArcFace 갤러리(등록자별 대표 임베딩) 생성

train.py처럼 CNN을 학습하는 게 아니라, 사전학습된 ArcFace 모델로
등록자별 사진들의 임베딩을 뽑아 평균낸 "대표 벡터"만 저장한다.

실행:  python enroll_arcface.py
       python enroll_arcface.py --limit 30   # 사람당 최대 30장만 랜덤 사용(처리 시간 단축)
"""
import argparse
import random

import numpy as np
from insightface.app import FaceAnalysis

import config
from utils import imread_unicode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="사람당 최대 사용 장수 (기본: 폴더의 전부 사용)")
    args = parser.parse_args()

    app = FaceAnalysis(name=config.ARCFACE_MODEL_NAME,
                        providers=["CPUExecutionProvider"],
                        allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    if not config.DATASET_DIR.exists():
        raise SystemExit(
            f"{config.DATASET_DIR} 폴더가 없습니다.\n"
            f"dataset/는 .gitignore에 걸려 있어 git으로 안 옮겨진다.\n"
            f"다른 PC에 있다면 그 dataset/ 폴더를 이 경로로 복사하거나,\n"
            f"없다면 main.py 실행 후 'r' 키로 사람별 사진을 다시 모아야 한다.\n"
            f"(ArcFace는 CNN 학습과 달리 인당 200장까지 필요 없고, "
            f"각도/표정 다양하게 10~20장이면 충분하다)"
        )

    person_dirs = sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir())
    if not person_dirs:
        raise SystemExit(f"{config.DATASET_DIR}에 등록자 폴더가 없습니다.")

    names, embeddings = [], []

    for person_dir in person_dirs:
        img_paths = list(person_dir.glob("*.jpg"))
        if args.limit and len(img_paths) > args.limit:
            img_paths = random.sample(img_paths, args.limit)
        vecs = []
        for path in img_paths:
            img = imread_unicode(path)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            # 사진 한 장에 얼굴이 여럿 잡히면 제일 큰 것 사용
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            vecs.append(face.normed_embedding)

        if not vecs:
            print(f"[스킵] {person_dir.name}: 임베딩 추출된 사진이 없음")
            continue

        proto = np.mean(vecs, axis=0)
        proto /= np.linalg.norm(proto)

        names.append(person_dir.name)
        embeddings.append(proto)
        print(f"[등록] {person_dir.name}: {len(vecs)}/{len(img_paths)}장 사용")

    if not names:
        raise SystemExit("등록된 임베딩이 하나도 없습니다.")

    np.savez(config.ARCFACE_GALLERY_PATH,
             names=np.array(names),
             embeddings=np.stack(embeddings))
    print(f"\n저장 완료: {config.ARCFACE_GALLERY_PATH} (인원 {len(names)}명)")


if __name__ == "__main__":
    main()

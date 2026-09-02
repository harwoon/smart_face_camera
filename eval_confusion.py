"""
저장된 모델을 평가해 confusion matrix와 리포트를 만든다.

실행:  python eval_confusion.py face_model.pth
       python eval_confusion.py results/stage1_base.pth --tag stage1_base

[ 왜 train.py에 안 넣고 따로 뺐나 ]
학습 중 val accuracy는 매 epoch 계산해야 하지만, confusion matrix는
'최종적으로 잘 나온 모델 하나'에 대해 한 번만 뽑으면 된다.
학습 루프에 넣으면 매 epoch마다 계산하는 낭비가 생긴다.

[ 4단계 비교 시 주의할 점 ]
train.py의 random_split은 시드 42로 고정되어 있지만, 데이터를
200장->400장으로 늘리면 전체 인덱스 개수가 달라져서 val set의
실제 이미지 구성도 달라진다. 즉 1단계와 2단계의 val accuracy는
완전히 같은 시험지로 비교한 게 아니다.

발표에서 엄밀한 비교를 하려면:
  1) 이 스크립트가 만드는 val 기반 결과는 "학습 중 진행 상황 확인용"으로 쓰고
  2) 4단계 전부 공통으로 쓸 별도의 '고정 테스트셋'(다른 날 찍은 사진)을
     따로 만들어서 --test-dir 옵션으로 넣어 비교하는 것을 권장한다.
     (config.DATASET_DIR와 다른 폴더에 미리 준비)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config
from train import build_model


def build_val_loader(test_dir=None):
    """test_dir가 주어지면 그 폴더 전체를 평가셋으로 쓴다.
    (여러 단계를 공정하게 비교하려면 이 방식을 권장)

    주어지지 않으면 train.py와 동일한 방식(같은 시드)으로
    학습에 쓰인 dataset에서 val 부분만 재구성한다.
    """
    val_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    if test_dir:
        ds = datasets.ImageFolder(test_dir, transform=val_tf)
        loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)
        return loader, ds.classes

    full_val = datasets.ImageFolder(config.DATASET_DIR, transform=val_tf)
    n_val = int(len(full_val) * config.VAL_RATIO)
    n_train = len(full_val) - n_val
    generator = torch.Generator().manual_seed(42)  # train.py와 동일한 시드
    _, val_idx = random_split(range(len(full_val)), [n_train, n_val],
                              generator=generator)
    val_set = torch.utils.data.Subset(full_val, list(val_idx))
    loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    return loader, full_val.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="평가할 .pth 파일 경로")
    parser.add_argument("--test-dir", default=None,
                        help="별도 고정 테스트셋 폴더 (없으면 학습 val split 사용)")
    parser.add_argument("--tag", default="eval",
                        help="결과 파일 이름에 붙일 태그 (예: stage1_base)")
    parser.add_argument("--outdir", default="results",
                        help="결과를 저장할 폴더")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model_path, map_location=device)
    ckpt_classes = ckpt["classes"]

    loader, data_classes = build_val_loader(args.test_dir)

    # 체크포인트의 클래스 순서와 평가 데이터의 클래스 순서가 다르면
    # confusion matrix 자체가 엉뚱하게 나온다. 반드시 일치를 확인한다.
    if ckpt_classes != data_classes:
        raise SystemExit(
            f"클래스 순서 불일치!\n"
            f"  모델 저장 시: {ckpt_classes}\n"
            f"  현재 데이터: {data_classes}\n"
            f"--test-dir가 학습에 쓰인 것과 다른 인원 구성이면 이 에러가 난다."
        )
    classes = ckpt_classes

    model = build_model(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()

    # ---------------------------------------------------- confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({args.tag})  acc={acc:.3f}")

    # 칸마다 숫자를 적어준다. 발표 슬라이드에서 이게 있어야 읽기 편하다.
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)

    fig.colorbar(im)
    fig.tight_layout()
    img_path = outdir / f"{args.tag}_confusion.png"
    fig.savefig(img_path, dpi=150)
    print(f"저장: {img_path}")

    # ---------------------------------------------------- classification report
    # confusion matrix보다 슬라이드에 넣기 더 간결한 표.
    # precision/recall/f1을 사람별로 보여줘서 "누가 잘 안 맞는지"가 한눈에 보인다.
    report = classification_report(all_labels, all_preds, target_names=classes)
    report_path = outdir / f"{args.tag}_report.txt"
    report_path.write_text(f"accuracy: {acc:.4f}\n\n{report}")
    print(f"저장: {report_path}")
    print(f"\n전체 정확도: {acc:.4f}\n")
    print(report)


if __name__ == "__main__":
    main()
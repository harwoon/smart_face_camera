"""
CNN 전이학습 (요구사항 5).

단독 실행:  python train.py

[ 절대 놓치면 안 되는 것 ]
1. dataset.classes를 모델과 함께 저장할 것.
   ImageFolder는 폴더명을 알파벳순으로 정렬해 인덱스를 매긴다.
   추론 쪽에서 os.listdir() 순서로 리스트를 다시 만들면 순서가 어긋나
   "이름만 서로 바뀐 채 잘 돌아가는" 최악의 버그가 난다.

2. train / val transform을 분리할 것.
   검증 데이터에 랜덤 증강이 들어가면 성능 측정이 무의미해진다.

3. Normalize는 ImageNet 값을 그대로 쓸 것.
   사전학습 가중치가 그 분포에 맞춰져 있다.
"""
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

import config
from utils import get_device


# ------------------------------------------------------------------

# 백본 선택
# BACKBONE = "resnet18"
BACKBONE = "mobilenet_v2"

UNFREEZE_LAST_BLOCK = False   # True로 바꾸면 마지막 블록까지 fine-tuning

# eval_confusion.py에서 결과 파일 이름에 쓸 태그
EXPERIMENT_TAG = f"{BACKBONE}" + ("_unfrozen" if UNFREEZE_LAST_BLOCK else "_base")


# ------------------------------------------------------------------ transform
def build_transforms():
    """학습용/검증용 transform을 만든다.

    데이터가 적으므로 증강은 적극적으로 쓰는 게 좋다.
    단, 검증용에는 절대 랜덤 증강을 넣지 말 것.
    """
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        # TODO: 증강을 더 추가해볼 것
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    return train_tf, val_tf


# ------------------------------------------------------------------ model
def build_model(num_classes, backbone=None):
    """사전학습 백본의 분류기만 교체해 전이학습.

    train.py에서는 backbone을 안 넘기면 위쪽 BACKBONE 상수를 그대로 쓴다.
    eval_confusion.py / recognizer.py는 체크포인트에 저장된 arch 값을
    명시적으로 넘겨서, train.py의 현재 BACKBONE 설정과 무관하게
    항상 올바른 구조로 모델을 재생성한다.

    ResNet18과 MobileNetV2는 구조 이름이 다르므로
    ("layer4"/"fc" vs "features[-1]"/"classifier") 서로 다른 코드가 필요하다.
    """
    backbone = backbone or BACKBONE

    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        if UNFREEZE_LAST_BLOCK:
            for param in model.layer4.parameters():
                param.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        if UNFREEZE_LAST_BLOCK:
            # MobileNetV2에는 layer4가 없다. features의 마지막 블록이
            # ResNet의 layer4와 같은 역할(가장 태스크 특화된 마지막 conv 단)을 한다.
            for param in model.features[-1].parameters():
                param.requires_grad = True
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    else:
        raise ValueError(f"지원하지 않는 backbone: {backbone}")

    return model


# ------------------------------------------------------------------ train loop
def run_epoch(model, loader, criterion, optimizer, device, train=True, scaler=None):
    model.train() if train else model.eval()

    use_amp = scaler is not None and device.type == "cuda"
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # AMP(자동 혼합 정밀도): float32 대신 float16으로 계산해
            # GPU 학습을 크게 앞당긴다. CPU에서는 효과가 없어 끈다.
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    print(f"실험 설정: backbone={BACKBONE}, unfreeze_last_block={UNFREEZE_LAST_BLOCK}")
    print(f"태그: {EXPERIMENT_TAG}")

    device = get_device()
    use_cuda = device.type == "cuda"

    train_tf, val_tf = build_transforms()

    # 같은 폴더를 transform만 다르게 두 번 읽고 인덱스로 나눈다.
    full_train = datasets.ImageFolder(config.DATASET_DIR, transform=train_tf)
    full_val = datasets.ImageFolder(config.DATASET_DIR, transform=val_tf)

    classes = full_train.classes
    print(f"클래스: {classes}")
    print(f"전체 이미지: {len(full_train)}장")
    if len(classes) < 2:
        raise SystemExit("최소 2명 이상의 데이터가 필요합니다.")

    n_val = int(len(full_train) * config.VAL_RATIO)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(len(full_train)),
                                      [n_train, n_val], generator=generator)

    train_set = torch.utils.data.Subset(full_train, list(train_idx))
    val_set = torch.utils.data.Subset(full_val, list(val_idx))

    # GPU를 쓸 때만 의미 있는 옵션들.
    #   num_workers : 이미지 로딩/증강을 별도 프로세스에서 병렬로. GPU가 굶지 않게 한다.
    #   pin_memory  : CPU->GPU 전송을 빠르게.
    # Windows에서 num_workers > 0이면 반드시 if __name__ == "__main__" 안에서
    # 실행해야 한다. (이 파일은 그렇게 되어 있다)
    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS if use_cuda else 0,
        pin_memory=use_cuda,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE)

    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    best_acc = 0.0
    t_start = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, True, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion,
                                    optimizer, device, False, scaler)
        print(f"[{epoch:02d}/{config.EPOCHS}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{time.time() - t0:.1f}s")

        if va_acc >= best_acc:
            best_acc = va_acc
            # classes를 반드시 함께 저장한다.
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "arch": BACKBONE,
                "img_size": config.IMG_SIZE,
                "face_margin": config.FACE_MARGIN,
            }, config.MODEL_PATH)

    print(f"\n저장 완료: {config.MODEL_PATH} "
          f"(best val acc {best_acc:.4f}, 총 {time.time() - t_start:.1f}s)")
    print("주의: val acc가 99%여도 좋아하지 말 것. 같은 자리에서 연속 촬영한")
    print("      데이터를 랜덤 split하면 train/val에 거의 같은 사진이 들어간다.")
    print("      진짜 검증은 다른 시간/다른 장소에서 새로 찍어서 해볼 것.")
    print()
    print("여러 실험(backbone/데이터추가/layer4/증강)을 비교할 계획이면")
    print("다음 실험을 돌리기 전에 반드시 이 파일을 다른 이름으로 복사해둘 것.")
    print(f"  cp {config.MODEL_PATH} results/{EXPERIMENT_TAG}.pth")
    print("안 하면 다음 python train.py 실행 시 덮어써져서 되돌릴 수 없다.")


if __name__ == "__main__":
    main()
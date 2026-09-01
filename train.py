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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

import config


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
def build_model(num_classes):
    """사전학습 ResNet18의 마지막 fc만 교체해 전이학습.

    TODO: 다른 백본도 실험해볼 것 (mobilenet_v2가 더 가벼워 실시간에 유리)
    TODO: 정확도가 아쉬우면 layer4까지 requires_grad=True로 풀어
          작은 learning rate로 fine-tuning 해볼 것
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ------------------------------------------------------------------ train loop
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE)

    best_acc = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion,
                                    optimizer, device, train=False)
        print(f"[{epoch:02d}/{config.EPOCHS}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc >= best_acc:
            best_acc = va_acc
            # classes를 반드시 함께 저장한다.
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "arch": "resnet18",
                "img_size": config.IMG_SIZE,
                "face_margin": config.FACE_MARGIN,
            }, config.MODEL_PATH)

    print(f"\n저장 완료: {config.MODEL_PATH} (best val acc {best_acc:.4f})")
    print("주의: val acc가 99%여도 좋아하지 말 것. 같은 자리에서 연속 촬영한")
    print("      데이터를 랜덤 split하면 train/val에 거의 같은 사진이 들어간다.")
    print("      진짜 검증은 다른 시간/다른 장소에서 새로 찍어서 해볼 것.")


if __name__ == "__main__":
    main()

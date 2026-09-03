"""
저장된 모델의 추론 속도(FPS)를 측정. 웹캠 없이 더미 이미지로 측정.

실행:  python bench_fps.py results/stage_layer4.pth
       python bench_fps.py results/mobilenet_final.pth --n 200

"""
import argparse
import time

import torch

import config
from train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--n", type=int, default=100, help="측정에 쓸 반복 횟수")
    parser.add_argument("--warmup", type=int, default=10, help="워밍업 반복 횟수")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.model_path, map_location=device)
    arch = ckpt.get("arch", "resnet18")
    classes = ckpt["classes"]

    model = build_model(len(classes), backbone=arch)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # 실제 웹캠 프레임과 같은 크기의 더미 입력 (얼굴 크롭 1장 기준)
    dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(args.n):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

    fps = args.n / elapsed
    ms_per_frame = elapsed / args.n * 1000

    print(f"\n모델: {args.model_path}  (arch={arch})")
    print(f"총 {args.n}회 추론: {elapsed:.3f}초")
    print(f"평균: {ms_per_frame:.2f} ms/frame")
    print(f"이론상 최대 FPS: {fps:.1f}")


if __name__ == "__main__":
    main()
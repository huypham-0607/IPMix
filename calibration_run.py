import argparse
import csv
import os

import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from calibration_tools import calib_err
from models.WideResNet_pytorch.wideresnet import WideResNet
from models.ResNeXt_DenseNet.models.resnext import resnext29
from models.ResNet.resnet import resnet18

PRETRAINED_DIR = "./pretrained/cifar"

# depth / widen_factor per wrn checkpoint name, matching the IPMix repo's
# --layers / --widen-factor defaults for wrn40-4 and the wrn28-10 variant.
WRN_CONFIGS = {
    "wrn28-10": {"depth": 28, "widen_factor": 10},
    "wrn40-4": {"depth": 40, "widen_factor": 4},
}


@torch.inference_mode()
def collect_calibration_data(model, loader, device="cuda"):
    model.eval()

    all_confidence = []
    all_correct = []
    all_pred = []
    all_target = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        # [batch, num_classes]
        logits = model(images)

        # Convert logits -> probabilities
        probs = torch.softmax(logits, dim=1)

        # Top predicted class and its probability
        confidence, pred = probs.max(dim=1)

        correct = pred.eq(targets)

        all_confidence.append(confidence.cpu())
        all_correct.append(correct.cpu())
        all_pred.append(pred.cpu())
        all_target.append(targets.cpu())

    confidence = torch.cat(all_confidence).numpy()
    correct = torch.cat(all_correct).numpy().astype(float)
    pred = torch.cat(all_pred).numpy()
    target = torch.cat(all_target).numpy()

    return confidence, correct, pred, target


def loader(dataset: str) -> DataLoader:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    if (dataset == 'cifar10') :
        test_set = datasets.CIFAR10(
            "./data/cifar",
            train=False,
            download=True,
            transform=test_transform
        )
    else :
        test_set = datasets.CIFAR100(
            "./data/cifar",
            train=False,
            download=True,
            transform=test_transform
        )

    test_loader = DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
        num_workers=4
    )

    return test_loader


def build_architecture(model: str, num_classes: int) -> torch.nn.Module:
    if model in WRN_CONFIGS:
        cfg = WRN_CONFIGS[model]
        return WideResNet(cfg["depth"], num_classes, cfg["widen_factor"], 0.0)
    if model == "resnext-29":
        return resnext29(num_classes=num_classes)
    if model == "resnet-18":
        # The IPMix repo's resnet18() drops the num_classes argument
        # (models/ResNet/resnet.py) so every checkpoint -- CIFAR-10 included --
        # was trained with a 100-way head. Match that shape or state_dict
        # loading will fail on the fc layer.
        return resnet18(num_classes=100)
    raise ValueError(f"Unknown model '{model}'. Expected one of "
                      f"{sorted([*WRN_CONFIGS, 'resnext-29', 'resnet-18'])}")


def load_model(dataset: str, model: str, device: str = "cuda") -> torch.nn.Module:
    num_classes = 10 if dataset == "cifar10" else 100

    net = build_architecture(model, num_classes)

    ckpt_path = os.path.join(PRETRAINED_DIR, dataset, f"{model}.pth.tar")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found for dataset={dataset!r}, "
                                 f"model={model!r} under {PRETRAINED_DIR}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Training wraps the net in nn.DataParallel before saving, so keys are
    # prefixed with "module.".
    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["state_dict"].items()
    }
    net.load_state_dict(state_dict)

    net = net.to(device)
    net.eval()
    return net

dataset_choices = ["cifar10", "cifar100"]
model_choices = [*WRN_CONFIGS, "resnext-29", "resnet-18"]

CSV_FIELDS = ["dataset", "model", "n_samples", "accuracy_pct", "rms_calib_error_pct"]


def run_single(dataset: str, model: str, device: str, out_dir: str) -> dict:
    net = load_model(dataset, model, device)
    test_loader = loader(dataset)

    confidence, correct, pred, target = collect_calibration_data(
        net, test_loader, device
    )

    accuracy = float(correct.mean())
    # Bit-for-bit reproduction of the repo's own (off-by-last-bin) metric.
    rms_repo = float(calib_err(confidence, correct, p="2"))

    print(f"{dataset:8s} {model:10s} "
          f"error {100 * (1 - accuracy):5.2f}% | RMS calib error {100 * rms_repo:5.2f}%")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f"{dataset}_{model}_calibration_raw.npz"),
        confidence=confidence,
        correct=correct,
        prediction=pred,
        target=target,
    )

    return {
        "dataset": dataset,
        "model": model,
        "n_samples": len(correct),
        "accuracy_pct": round(100 * accuracy, 2),
        "rms_calib_error_pct": round(100 * rms_repo, 2),
    }


def write_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def run_single_cmd(args):
    row = run_single(args.dataset, args.model, args.device, args.out_dir)
    write_csv([row], os.path.join(
        args.out_dir, f"{args.dataset}_{args.model}_calibration.csv"))


def run_all_cmd(args):
    rows = [
        run_single(dataset, model, args.device, args.out_dir)
        for dataset in dataset_choices
        for model in model_choices
    ]
    write_csv(rows, os.path.join(args.out_dir, "calibration_summary.csv"))


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--out-dir", default="./result")

    parser = argparse.ArgumentParser(description="IPMix calibration reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser(
        "run-single", parents=[common],
        help="Run calibration eval on one (dataset, model) pair.")
    single_parser.add_argument("--dataset", choices=dataset_choices, default="cifar100")
    single_parser.add_argument("--model", choices=model_choices, default="wrn40-4")
    single_parser.set_defaults(func=run_single_cmd)

    all_parser = subparsers.add_parser(
        "run-all", parents=[common],
        help="Run calibration eval on every (dataset, model) pair.")
    all_parser.set_defaults(func=run_all_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

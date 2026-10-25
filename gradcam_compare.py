import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import datasets, transforms

from config import TEST_DIR, FINETUNE_CHECKPOINTS, OUT_46, MAIN_CODE_ROOT

import sys

if str(MAIN_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_CODE_ROOT))

from finetune_model import ResNet50WithCBAM

STRATEGY_NAMES = {
    "frozen": "FT-A Frozen Backbone",
    "layer4": "FT-B Partial Fine-Tuning",
    "full": "FT-C Full Fine-Tuning"
}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self.save_gradient)

    def save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, input_tensor):
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        pred_idx = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[pred_idx].item())

        score = output[:, pred_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam, pred_idx, confidence

    def close(self):
        self.handle.remove()

def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    model = ResNet50WithCBAM(num_classes=2)

    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint tidak cocok. Missing={result.missing_keys}, Unexpected={result.unexpected_keys}")

    model = model.to(device)
    model.eval()

    return model

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tensor = transform(image).unsqueeze(0)
    display = np.asarray(image.resize((224, 224))).astype(np.float32) / 255.0

    return tensor, display

def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probability = F.softmax(output, dim=1)[0]
        pred = int(torch.argmax(probability).item())
        confidence = float(probability[pred].item())

    return pred, confidence

def case_name(true_idx, pred_idx, normal_idx, abnormal_idx):
    if true_idx == abnormal_idx and pred_idx == abnormal_idx:
        return "TP"

    if true_idx == normal_idx and pred_idx == normal_idx:
        return "TN"

    if true_idx == abnormal_idx and pred_idx == normal_idx:
        return "FN"

    return "FP"

def overlay_cam(image, cam):
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.60 * image + 0.40 * heatmap, 0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=["frozen", "layer4", "full"])
    parser.add_argument("--per-case", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_path = FINETUNE_CHECKPOINTS["baseline"]
    proposed_path = FINETUNE_CHECKPOINTS[args.strategy]

    print(f"Device        : {device}")
    print(f"Baseline      : {baseline_path}")
    print(f"Proposed      : {proposed_path}")
    print(f"Strategy      : {STRATEGY_NAMES[args.strategy]}")

    baseline = load_model(baseline_path, device)
    proposed = load_model(proposed_path, device)

    dataset = datasets.ImageFolder(TEST_DIR)

    class_to_idx = dataset.class_to_idx
    idx_to_class = {value: key for key, value in class_to_idx.items()}

    normal_idx = class_to_idx["Normal"]
    abnormal_idx = class_to_idx["Abnormal"]

    selected = {
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": []
    }

    for image_path, true_idx in dataset.samples:
        tensor, _ = load_image(image_path)
        tensor = tensor.to(device)

        proposed_pred, proposed_conf = predict(proposed, tensor)
        baseline_pred, baseline_conf = predict(baseline, tensor)

        category = case_name(true_idx, proposed_pred, normal_idx, abnormal_idx)

        if len(selected[category]) < args.per_case:
            selected[category].append({
                "path": image_path,
                "true": true_idx,
                "baseline_pred": baseline_pred,
                "baseline_conf": baseline_conf,
                "proposed_pred": proposed_pred,
                "proposed_conf": proposed_conf
            })

        if all(len(values) >= args.per_case for values in selected.values()):
            break

    baseline_cam = GradCAM(baseline, baseline.layer4[-1])
    proposed_cam = GradCAM(proposed, proposed.layer4[-1])

    output_dir = OUT_46 / args.strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    for category, samples in selected.items():
        for index, sample in enumerate(samples, start=1):
            tensor, display = load_image(sample["path"])
            tensor = tensor.to(device)

            cam_baseline, baseline_pred, baseline_conf = baseline_cam.generate(tensor)
            cam_proposed, proposed_pred, proposed_conf = proposed_cam.generate(tensor)

            baseline_overlay = overlay_cam(display, cam_baseline)
            proposed_overlay = overlay_cam(display, cam_proposed)

            true_name = idx_to_class[sample["true"]]
            baseline_name = idx_to_class[baseline_pred]
            proposed_name = idx_to_class[proposed_pred]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(display)
            axes[0].set_title(f"Original\nTrue: {true_name}")

            axes[1].imshow(baseline_overlay)
            axes[1].set_title(f"Baseline ImageNet + CBAM\nPred: {baseline_name}\nConf: {baseline_conf:.4f}")

            axes[2].imshow(proposed_overlay)
            axes[2].set_title(f"{STRATEGY_NAMES[args.strategy]}\nPred: {proposed_name}\nConf: {proposed_conf:.4f}")

            for ax in axes:
                ax.axis("off")

            plt.suptitle(f"{category} - {sample['path'].split(chr(92))[-1]}")
            plt.tight_layout()

            output_path = output_dir / f"4_6_{category}_{index}.png"

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"[Saved] {output_path}")

    baseline_cam.close()
    proposed_cam.close()

    print("\nGrad-CAM comparison selesai.")

if __name__ == "__main__":
    main()
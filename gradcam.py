import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms

from config import EXTERNAL_DRIVE
from finetune_model import ResNet50WithCBAM

MODEL_NAME = "ResNet50_USULAN_Dengan_SimCLR_dan_CBAM"
MODEL_PATH = os.path.join(EXTERNAL_DRIVE, "hasil_finetune_simclr", f"{MODEL_NAME}_finetuned_best.pth")
TEST_DIR = os.path.join(EXTERNAL_DRIVE, "test")
OUTPUT_DIR = os.path.join(EXTERNAL_DRIVE, "hasil_finetune_simclr", "gradcam")
NUM_IMAGES_PER_CLASS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        probabilities = F.softmax(output.detach(), dim=1)[0]
        prediction = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[prediction].item())

        return cam, prediction, confidence

    def close(self):
        self.handle.remove()

def get_class_index(class_to_idx, target):
    for name, idx in class_to_idx.items():
        if name.lower() == target.lower():
            return idx
    raise RuntimeError(f"Kelas {target} tidak ditemukan: {class_to_idx}")

def load_model(device):
    model = ResNet50WithCBAM(num_classes=2)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint fine-tuning tidak cocok. Missing={result.missing_keys}, Unexpected={result.unexpected_keys}")

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

    input_tensor = transform(image).unsqueeze(0)

    display_image = image.resize((224, 224))
    display_image = np.asarray(display_image).astype(np.float32) / 255.0

    return input_tensor, display_image

def save_gradcam(original, cam, output_path, true_name, pred_name, confidence):
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.60 * original + 0.40 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nTrue: {true_name} | Pred: {pred_name}\nConfidence: {confidence:.4f}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device    : {device}")
    print(f"Model     : {MODEL_PATH}")
    print(f"Test data : {TEST_DIR}")

    dataset = datasets.ImageFolder(TEST_DIR)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"Class mapping: {class_to_idx}")

    model = load_model(device)

    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    class_counts = {idx: 0 for idx in idx_to_class}

    for image_path, true_idx in dataset.samples:
        if class_counts[true_idx] >= NUM_IMAGES_PER_CLASS:
            continue

        input_tensor, display_image = load_image(image_path)
        input_tensor = input_tensor.to(device)

        cam, pred_idx, confidence = gradcam.generate(input_tensor)

        true_name = idx_to_class[true_idx]
        pred_name = idx_to_class[pred_idx]

        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_name = f"{true_name}_{filename}_pred_{pred_name}.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        save_gradcam(
            display_image,
            cam,
            output_path,
            true_name,
            pred_name,
            confidence
        )

        class_counts[true_idx] += 1

        print(f"[Grad-CAM] {filename} | True={true_name} | Pred={pred_name} | Confidence={confidence:.4f}")

        if all(count >= NUM_IMAGES_PER_CLASS for count in class_counts.values()):
            break

    gradcam.close()

    print(f"\nGrad-CAM selesai. Output: {OUTPUT_DIR}")
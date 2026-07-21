import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

from config import RAW_DATA_DIR, CHECKPOINT_DIR
from finetune import get_model, get_finetune_transforms

def plot_confusion_matrix(device, model_path):
    print("Membangun Confusion Matrix dari Data Validasi...")
    
    model = get_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Memuat data validasi murni (20%) dengan seed yang sama seperti finetune.py
    val_full = datasets.ImageFolder(root=RAW_DATA_DIR, transform=get_finetune_transforms(is_training=False))
    train_size = int(0.8 * len(val_full))
    val_size = len(val_full) - train_size
    
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(val_full, [train_size, val_size], generator=generator)
    
    dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Plotting
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=val_full.classes, 
                yticklabels=val_full.classes)
    plt.title('Confusion Matrix - Hasil Klasifikasi Kanker Serviks (Data Validasi)')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Prediksi Model')
    plt.show()

def generate_gradcam(device, model_path, image_path, target_class=1):
    print(f"Menganalisis gambar menggunakan Grad-CAM: {image_path}")
    
    model = get_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rgb_img_pil = Image.open(image_path).convert('RGB').resize((224, 224))
    rgb_img_np = np.float32(rgb_img_pil) / 255.0
    
    transform = get_finetune_transforms(is_training=False)
    input_tensor = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(target_class)] 
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
    
    #Tampilkan
    plt.figure(figsize=(6,6))
    plt.imshow(visualization)
    plt.title(f'Grad-CAM (Target Class: {target_class})')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model_path = os.path.join(CHECKPOINT_DIR, "final_classifier_model.pth")
    
    if os.path.exists(final_model_path):
        # Jalankan salah satu atau keduanya (bisa di-comment jika tidak perlu)
        plot_confusion_matrix(device, final_model_path)
        
        # Contoh penggunaan Grad-CAM (Ganti path ke gambar serviks acak dari dataset Anda)
        sample_image = os.path.join(RAW_DATA_DIR, "Type_1", "10.jpg") # Sesuaikan nama file
        if os.path.exists(sample_image):
            generate_gradcam(device, final_model_path, sample_image, target_class=0)
    else:
        print("Model belum dilatih! Jalankan finetune.py terlebih dahulu.")
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import shutil


class OASISFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = [
            "Mild_Dementia",
            "Moderate_Dementia",
            "Non_Demented",
            "Very_Mild_Dementia",
        ]
        self.image_paths = []
        self.labels = []

        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image_pil = Image.open(img_path).convert("L")
        img_array = np.array(image_pil)
        img_norm = img_array.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_norm).unsqueeze(0)
        return img_tensor, label


def generate_preprocessed_dataset(raw_dir, save_dir):
    abs_raw_dir = os.path.abspath(raw_dir)
    abs_save_dir = os.path.abspath(save_dir)

    print("Raw directory:", abs_raw_dir)
    print("Exists:", os.path.exists(abs_raw_dir))

    if not os.path.exists(abs_raw_dir):
        return

    if os.path.exists(abs_save_dir):
        shutil.rmtree(abs_save_dir)
    os.makedirs(abs_save_dir)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    classes = [
        "Mild_Dementia",
        "Moderate_Dementia",
        "Non_Demented",
        "Very_Mild_Dementia",
    ]

    for class_name in classes:
        raw_class_dir = os.path.join(abs_raw_dir, class_name)
        save_class_dir = os.path.join(abs_save_dir, class_name)

        if not os.path.exists(raw_class_dir):
            continue

        os.makedirs(save_class_dir, exist_ok=True)
        images = [
            f
            for f in os.listdir(raw_class_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]

        for img_name in images:
            raw_path = os.path.join(raw_class_dir, img_name)
            save_path = os.path.join(save_class_dir, img_name)

            try:
                image_pil = Image.open(raw_path).convert("L")
                img_array = np.array(image_pil)
                img_resized = cv2.resize(
                    img_array, (128, 128), interpolation=cv2.INTER_AREA
                )
                img_clahe = clahe.apply(img_resized)
                cv2.imwrite(save_path, img_clahe)
            except Exception:
                print(f"Failed to process {raw_path}: {e}")


if __name__ == "__main__":
    MY_RAW_DATA_FOLDER = "./backend/Data"
    MY_NEW_SAVE_FOLDER = "./Processed_Data"
    generate_preprocessed_dataset(
        raw_dir=MY_RAW_DATA_FOLDER, save_dir=MY_NEW_SAVE_FOLDER
    )

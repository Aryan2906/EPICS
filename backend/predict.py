import torch
import cv2
import numpy as np
from PIL import Image
from snntorch import utils
import sys
import os

sys.path.append(os.path.abspath("../backend"))
from model import CNNtoSNNConverter
from train import DeepSNNClassifier


def predict_mri(
    image_path, retina_weights="snn_retina.pth", brain_weights="snn_brain.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        image_pil = Image.open(image_path).convert("L")
        img_array = np.array(image_pil)
        img_resized = cv2.resize(img_array, (128, 128), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_resized)
        img_norm = img_clahe.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    except Exception:
        return

    retina = CNNtoSNNConverter().to(device)
    brain = DeepSNNClassifier().to(device)

    retina.load_state_dict(
        torch.load(retina_weights, map_location=device, weights_only=True)
    )
    brain.load_state_dict(
        torch.load(brain_weights, map_location=device, weights_only=True)
    )

    retina.eval()
    brain.eval()

    classes = [
        "Mild_Dementia",
        "Moderate_Dementia",
        "Non_Demented",
        "Very_Mild_Dementia",
    ]
    num_steps = 15
    spk_rec = []

    with torch.no_grad():
        utils.reset(retina)
        utils.reset(brain)

        for step in range(num_steps):
            spikes_2048, _ = retina(img_tensor)
            class_spikes, _ = brain(spikes_2048)
            spk_rec.append(class_spikes)

        spk_rec = torch.stack(spk_rec)
        total_spikes = spk_rec.sum(dim=0).squeeze(0)

    raw_counts = total_spikes.cpu().numpy()

    if raw_counts.sum() != 0:
        predicted_idx = raw_counts.argmax()
        confidence = (raw_counts[predicted_idx] / raw_counts.sum()) * 100
        print(f"Predicted Pattern  : {classes[predicted_idx]}")
        print(f"Model Confidence   : {confidence:.2f}%")


if __name__ == "__main__":
    test_image = "image.jpg"
    predict_mri(test_image)

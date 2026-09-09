from model import CNNtoSNNConverter
from train import DeepSNNClassifier
import torch

import cv2
import numpy as np
from PIL import Image
from snntorch import utils
import sys
import os
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv
import streamlit as st


load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
retina_path = os.path.join(BASE_DIR, "snn_retina.pth")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_patient_report(heatmap_path, snn_diagnosis, confidence,heatmap_features):
    try:
        base64_image = encode_image(heatmap_path)
    except FileNotFoundError:
        return

    system_prompt = """
            You are an experienced neurologist and AI explainability assistant.
    
            You will receive:
            1. A brain MRI Grad-CAM heatmap.
            2. The predicted dementia class.
            3. The model confidence.
    
            The heatmap is an overlay where:
            - Red = highest model attention.
            - Yellow = high attention.
            - Green = moderate attention.
            - Blue = low attention.
    
            Your job is to carefully inspect the image and explain:
    
            1. What regions appear highlighted.
            2. Whether those regions correspond to cortical atrophy, enlarged ventricles, hippocampal shrinkage, or other visible structural patterns.
            3. Why those highlighted regions could support the AI prediction.
            4. Whether the highlighted regions appear localized or widespread.
            5. Explain the confidence score in simple language.
            6. End with a disclaimer that this is an AI-assisted screening tool and not a medical diagnosis.
    
            Do NOT simply describe the colors.
            Interpret what they indicate anatomically.
            Write professionally but in language understandable to patients.
        """

    human_prompt = f"""
    Prediction:
    {snn_diagnosis}

    Confidence:
    {confidence:.2f}%

    Explainability Metrics

    Activation Area:
    {heatmap_features['activation_percentage']:.2f}%

    Peak Activation:
    {heatmap_features['peak_activation']:.3f}

    Dominant Hemisphere:
    {heatmap_features['dominant_hemisphere']}

    Peak Coordinate:
    ({heatmap_features['peak_x']},
    {heatmap_features['peak_y']})

    Explain how these explainability metrics support the AI prediction.
    """
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=2500,
            ),
            contents=[
                human_prompt,
                types.Part.from_bytes(
                    data=open(heatmap_path, "rb").read(),
                    mime_type="image/jpeg",
                ),
            ],
        )

        report = response.text
        print(report)
        print(response.candidates[0].finish_reason)

        st.subheader("AI Generated Patient Report")
        st.write(report)
        with open("final_patient_report.txt", "w") as f:
            f.write(report)
    except Exception as e:
        print(f"API Communication Failed: {e}")


def generate_heatmap_and_report(
    image_path, retina_weights="snn_retina.pth", brain_weights="snn_brain.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        image_pil = Image.open(image_path).convert("L")
        img_array = np.array(image_pil)
        img_resized = cv2.resize(img_array, (128, 128), interpolation=cv2.INTER_AREA)
        original_bg = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_resized)
        img_norm = img_clahe.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).to(device)
        img_tensor.requires_grad = True
    except Exception as e:
        print(f"Failed to load image: {e}")
        return
    retina_path = os.path.join(BASE_DIR, retina_weights)
    brain_path = os.path.join(BASE_DIR, brain_weights)
    retina = CNNtoSNNConverter().to(device)
    brain = DeepSNNClassifier().to(device)
    retina.load_state_dict(
        torch.load(retina_path, map_location=device, weights_only=True)
    )
    brain.load_state_dict(
        torch.load(brain_path, map_location=device, weights_only=True)
    )

    retina.eval()
    brain.eval()

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    target_layer = retina.cnn_extractor[5]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    classes = [
        "Mild_Dementia",
        "Moderate_Dementia",
        "Non_Demented",
        "Very_Mild_Dementia",
    ]
    spk_rec = []

    utils.reset(retina)
    utils.reset(brain)

    for step in range(50):
        spikes_2048, _ = retina(img_tensor)
        class_spikes, _ = brain(spikes_2048)
        spk_rec.append(class_spikes)

    spk_rec = torch.stack(spk_rec)
    total_spikes = spk_rec.sum(dim=0).squeeze(0)

    predicted_idx = total_spikes.argmax()
    print("Total spikes:", total_spikes)
    print("Sum:", total_spikes.sum())
    confidence = (total_spikes[predicted_idx] / total_spikes.sum()) * 100
    diagnosis = classes[predicted_idx]

    st.success("Analysis Complete")
    st.write(f"### Diagnosis: {diagnosis}")
    st.write(f"### Confidence: {confidence:.2f}%")

    brain.zero_grad()
    retina.zero_grad()
    target_score = total_spikes[predicted_idx]
    target_score.backward()

    if activations is not None and gradients is not None:
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.resize(heatmap, (128, 128))
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_bg, 0.6, heatmap_colored, 0.4, 0)

        #heatmaps feature extraction
        threshold = 0.6
        mask = (heatmap > threshold).astype(np.uint8)

        activation_percentage = np.sum(mask) / mask.size * 100

        peak = np.max(heatmap)

        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        left = np.sum(heatmap[:, :heatmap.shape[1]//2])
        right = np.sum(heatmap[:, heatmap.shape[1]//2:])

        dominant = "Left" if left > right else "Right"

        heatmap_features = {
            "activation_percentage": activation_percentage,
            "peak_activation": peak,
            "peak_x": int(x),
            "peak_y": int(y),
            "dominant_hemisphere": dominant,
        }
        

        output_filename = "brain_atrophy_analysis.jpg"
        cv2.imwrite(output_filename, superimposed_img)
        st.image(
            output_filename,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

        generate_patient_report(output_filename, diagnosis, confidence,heatmap_features)
    else:
        print(" Failed to extract gradients. Check network architecture hooks.")


st.title("🧠 NeuroVision AI")

uploaded_file = st.file_uploader(
    "Upload MRI Scan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    st.image(
        uploaded_file,
        caption="Uploaded MRI",
        use_container_width=True
    )

    if st.button("Analyze MRI"):

        with open("uploaded_scan.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        generate_heatmap_and_report("uploaded_scan.jpg")

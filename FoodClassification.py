import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="10-Class Food Classifier", layout="centered")
st.title("Food Classification (10 Classes)")
st.write("Upload an image to classify it into one of the 10 food categories.")

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Your 10 Class Names
# ---------------------------
class_names = [
    'beef_tartare',
    'cannoli',
    'ceviche',
    'chocolate_mousse',
    'clam_chowder',
    'crab_cakes',
    'dumplings',
    'foie_gras',
    'french_onion_soup',
    'frozen_yogurt'
]

# ---------------------------
# Load Model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)  # Using ResNet50 architecture
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)      # 10-class classifier

    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.to(device)
        model.eval()
        st.write("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e

    return model

model = load_model()

# ---------------------------
# Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Upload + Predict
# ---------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)          # logits for 10 classes
            probs = torch.softmax(outputs, dim=1)  # convert to probabilities
            
            pred_class_idx = torch.argmax(probs, dim=1).item()
            pred_class = class_names[pred_class_idx]
            confidence = probs[0, pred_class_idx].item() * 100

        st.success(f"**Prediction:** {pred_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

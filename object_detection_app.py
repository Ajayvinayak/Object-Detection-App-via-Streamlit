import tracemalloc
tracemalloc.start()

import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

# Function to set background and font styles
def set_styles():
    st.markdown(
        """
        <style>
        /* Set the background color for the entire page */
        .stApp {
            background-color: #808000; /* Olive Green */
        }

        /* Set all text to white */
        html, body, h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown {
            color: #0000000 !important; 
        }

        /* Customize buttons */
        .stButton button {
            font-family: 'Tahoma', sans-serif;
            font-size: 16px;
            background-color: #3498DB; 
            color: #FFFFFF !important; 
            border-radius: 8px;
            border: none;
        }

        /* Remove unnecessary Streamlit borders */
        .block-container {
            padding: 2rem; /* Add padding for better appearance */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to apply styles
set_styles()

# Load weights and model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

# Prediction function
def make_prediction(img): 
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

# Draw bounding boxes
def create_image_with_bboxes(img, prediction): 
    labels_with_scores = [
        f"{label} ({int(score * 100)}%)"
        for label, score in zip(prediction["labels"], prediction["scores"])
    ]
    
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor,
        boxes=prediction["boxes"],
        labels=labels_with_scores,
        colors=["yellow" for _ in prediction["labels"]],
        width=2
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

# Streamlit UI
st.title("Object Detector :tea: :coffee:")
upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)

    prediction = make_prediction(img)
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]
    prediction["scores"] = [f"{round(score * 100, 2)}%" for score in prediction["scores"].detach().numpy()]
    st.header("Predicted Probabilities")
    st.write(prediction)
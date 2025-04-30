import streamlit as st
import cv2
import numpy as np
from pixel_utils import generate_pixel_art

st.set_page_config(
    page_title="PixelCraft - Pixel Art Generator",
    layout="centered",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("Settings")
    num_colors = st.slider(
        "Color Count",
        min_value=4,
        max_value=16,
        value=8,
        help="Recommended: 8-16 colors for retro style"
    )
    pixel_size = st.slider(
        "Pixel Size",
        min_value=2,
        max_value=32,
        value=4,
        step=2,
        help="Size of each pixel block"
    )
    enhance_contrast = st.checkbox(
        "Enhance Contrast",
        value=False,
        help="Improve color vibrancy"
    )

# Main interface
st.title("PixelCraft - Pixel Art Generator")
st.markdown("Transform images into retro pixel art instantly!")
# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image...",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG/PNG formats (max 5MB)"
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Processing
    with st.spinner("Generating pixel art..."):
        processed_image = generate_pixel_art(
            original_image,
            num_colors=num_colors,
            pixel_size=pixel_size
        )
        if enhance_contrast:
            processed_image = cv2.convertScaleAbs(
                processed_image,
                alpha=1.2,
                beta=20
            )

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, channels="BGR", caption="Original Image")
    with col2:
        st.image(processed_image, channels="BGR", caption="Pixel Art")

    # Download button
    _, encoded_image = cv2.imencode(".png", processed_image)
    st.download_button(
        label="Download Result",
        data=encoded_image.tobytes(),
        file_name="pixel_art.png",
        mime="image/png"
    )


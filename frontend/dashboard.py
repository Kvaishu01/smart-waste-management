# frontend/dashboard.py

import streamlit as st
from PIL import Image
import requests

st.set_page_config(page_title="Smart Waste Management Dashboard", layout="wide")

st.title("🚮 Smart Waste Management Dashboard")
st.markdown("Welcome to the Smart Waste Management system. Upload an image to classify the type of waste.")

# Upload and preview image
uploaded_file = st.file_uploader("Upload a waste image for classification", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Waste Image", use_container_width=True)

    # Send image to backend for prediction
    with st.spinner("Classifying..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict/",
                files={"file": uploaded_file.getvalue()}
            )

            if response.status_code == 200:
                prediction = response.json()

                # ✅ Check and show predicted class
                if "class" in prediction and "confidence" in prediction:
                    st.success(
                        f"🧾 **Predicted Class:** `{prediction['class'].capitalize()}`\n\n"
                        f"🎯 **Confidence:** `{prediction['confidence']:.2f}`"
                    )
                elif "error" in prediction:
                    st.error(f"⚠️ Backend Error: {prediction['error']}")
                else:
                    st.error("⚠️ Unexpected response format from backend.")
            else:
                st.error(f"❌ HTTP Error {response.status_code}: Backend did not respond correctly.")
        except Exception as e:
            st.error(f"❌ Request failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and FastAPI")

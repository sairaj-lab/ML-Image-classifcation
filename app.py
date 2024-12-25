import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from .env file
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key is not set. Please configure the .env file correctly.")
    st.stop()

# Streamlit app setup
st.title("Image Classification with Gemini API")
st.write("Upload an image to classify it using a Gemini ML model.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated here
    
    # Convert image to bytes for API call
    image_bytes = uploaded_file.read()

    # Set up the Gemini API call
    gemini_model = "gemini-1.5-flash"  # Replace with the actual model name if different
    url = f"https://api.gemini.openai.com/v1/models/{gemini_model}/classify"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/octet-stream",
    }

    st.write("Classifying the image...")
    try:
        # Make the API request
        response = requests.post(url, headers=headers, data=image_bytes)
        
        if response.status_code == 200:
            # Parse and display the response
            result = response.json()
            st.write("Classification Result:")
            st.json(result)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.write("Make sure the Gemini API is properly configured for image classification.")

import streamlit as st
from PIL import Image
import requests
import io
import json
import base64

# Your Hugging Face API token (Read access)
HF_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/naver-clova-ix/donut-base-finetuned-docvqa"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}


def query(image_bytes, question):
    # Convert image to base64
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # Create payload with base64 encoded image
    payload = {
        "inputs": {
            "image": encoded_image,
            "question": question
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        # Print response details for debugging
        st.write(f"Status Code: {response.status_code}")

        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request error: {str(e)}", "details": str(e)}
    except json.JSONDecodeError:
        return {"error": f"Failed to parse API response as JSON. Status code: {response.status_code}",
                "raw_response": response.text}


# Streamlit App
st.set_page_config(page_title="🧠 Visual Insight Assistant", layout="centered")
st.title("🧠 Visual Insight Assistant (Donut Model)")

# Add API documentation link
st.markdown("""
This app uses the [Donut Document Visual Question Answering model](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa) 
to answer questions about document images. Created by Vikrant & Vishvali 
""")

uploaded_file = st.file_uploader("Upload an image (document screenshot or scan)", type=["jpg", "jpeg", "png"])
question = st.text_input("Ask a question about the document:")

if uploaded_file and question:
    image = Image.open(uploaded_file)

    # Resize image if it's too large
    max_size = 1000
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    # Show the uploaded image
    st.image(image, caption="Uploaded Document", use_column_width=True)

    # Convert image to bytes - Handle RGBA (PNG) images by converting to RGB
    buffered = io.BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="PNG")  # Using PNG format to avoid JPEG issues
    image_bytes = buffered.getvalue()

    with st.spinner("Analyzing document..."):
        result = query(image_bytes, question)

    st.markdown("### 💬 Response:")

    if isinstance(result, dict) and "error" in result:
        st.error(result["error"])
        if "details" in result:
            with st.expander("Error Details"):
                st.text(result["details"])
        if "raw_response" in result:
            with st.expander("Raw API Response"):
                st.text(result["raw_response"])
    else:
        # Handle the response properly based on its type
        if isinstance(result, list):
            if result and len(result) > 0:
                # If it's a list, display the first item or all items
                if isinstance(result[0], dict) and "answer" in result[0]:
                    st.success(result[0]["answer"])
                else:
                    st.success(str(result[0]))

                if len(result) > 1:
                    with st.expander("All Results"):
                        st.json(result)
            else:
                st.warning("The model returned an empty list.")
        elif isinstance(result, dict):
            # If it's a dictionary, try to extract the answer or display the whole dict
            answer = result.get("answer", None)
            if answer:
                st.success(answer)
            else:
                st.json(result)
        else:
            # For any other type of response
            st.success(str(result))

    # Add debugging section
    with st.expander("Debug Information"):
        st.write("Image size:", image.size)
        st.write("Image format:", image.format)
        st.write("Image mode:", image.mode)
        st.write("Response type:", type(result).__name__)
        st.write("Response structure:", result)

else:
    st.info("Upload a document image and ask a question to get started.")

    # Example section
    with st.expander("Example Usage"):
        st.markdown("""
        1. Upload a document, receipt, or form image
        2. Ask questions like:
           - "What is the total amount?"
           - "What is the invoice number?"
           - "Who is the recipient?"
        """)

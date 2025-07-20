import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import easyocr
from inference_sdk import InferenceHTTPClient
import cv2
import re

def uniform_format(plates):
    formatted = []
    for plate in plates:
        # Remove 'IND' from anywhere (case-insensitive)
        plate_no_ind = re.sub(r'ind', '', plate, flags=re.IGNORECASE)
        # Remove all non-alphanumeric characters
        plate_alphanum = re.sub(r'[^A-Za-z0-9]', '', plate_no_ind)
        # Convert to uppercase
        formatted.append(plate_alphanum.upper())
    return formatted

def check_authorization(plate, authorized_df):
    # plate is a single string, authorized_df has 'NumberPlate' column
    authorized_plates = uniform_format(authorized_df['NumberPlate'].astype(str).tolist())
    plate_uniform = uniform_format([plate])[0]
    return "Authorized" if plate_uniform in authorized_plates else "Unauthorized"

def ocr(img_np):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(img_np)
    text = " ".join([text for (_, text, _) in result])
    return text.strip()

st.title('Vehicle Number Plate OCR & Authorization')

# Secure API key from Streamlit secrets
api_key = st.secrets["ROBOFLOW_API_KEY"]
MODEL_ID = "vehicle-registration-plates-trudk/2"  # Fixed model

# Upload Excel or CSV with authorized number plates
auth_file = st.sidebar.file_uploader(
    "Upload Excel or CSV file of authorized plates (column name: 'NumberPlate')",
    type=['xlsx', 'csv']
)

uploaded_file = st.file_uploader("Upload vehicle image", type=['jpg', 'jpeg', 'png'])

# Storage for OCR result
extracted_plate = ""

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)[:, :, ::-1]  # PIL RGB → OpenCV BGR

    st.image(image, caption='Uploaded Image', use_container_width=True)

    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    with st.spinner("Detecting number plate..."):
        result = CLIENT.infer(img_np, model_id=MODEL_ID)
    
    if result and result.get('predictions'):
        prediction = result['predictions'][0]
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        x1, y1 = int(x - width/2), int(y - height/2)
        x2, y2 = int(x + width/2), int(y + height/2)
        cropped_img = img_np[y1:y2, x1:x2]

        display_img = img_np.copy()
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0,255,0), 2)
        st.image(display_img[:, :, ::-1], caption='Detected Number Plate', use_container_width=True)

        # OCR on cropped number plate (returns a single string)
        extracted_plate = ocr(cropped_img)
        st.session_state['extracted_plate'] = extracted_plate

        if extracted_plate:
            st.success(f"OCR Detected Plate: {extracted_plate}")
            st.image(cropped_img[:, :, ::-1], caption='Cropped Number Plate', use_container_width=False)
        else:
            st.warning("No text detected in number plate.")
    else:
        st.warning("No number plate detected in this image.")
else:
    if 'extracted_plate' in st.session_state:
        del st.session_state['extracted_plate']

if auth_file:
    # Load authorized number plates
    if auth_file.name.endswith('.xlsx'):
        auth_df = pd.read_excel(auth_file)
    else:
        auth_df = pd.read_csv(auth_file)
    
    # Check detected plate if available
    extracted_plate = st.session_state.get('extracted_plate', "")
    if extracted_plate:
        status = check_authorization(extracted_plate, auth_df)
        if status == "Authorized":
            st.markdown(
                '<span style="color:green; font-size:2em; font-weight:bold;">AUTHORIZED</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span style="color:red; font-size:2em; font-weight:bold;">UNAUTHORIZED</span>',
                unsafe_allow_html=True
            )
    
    st.markdown("**Not satisfied with the detection? Enter number plate for manual check:**")
    manual_plate = st.text_input("Manual Input Plate (single plate):", "")
    if manual_plate:
        manual_status = check_authorization(manual_plate, auth_df)
        if manual_status == "Authorized":
            st.markdown(
                '<span style="color:green; font-size:2em; font-weight:bold;">AUTHORIZED</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span style="color:red; font-size:2em; font-weight:bold;">UNAUTHORIZED</span>',
                unsafe_allow_html=True
            )
else:
    st.info("Please upload an Excel or CSV file of authorized number plates in the sidebar.")


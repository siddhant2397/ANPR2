import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import easyocr
from inference_sdk import InferenceHTTPClient
import cv2

# Utility to convert number plates to a uniform format (uppercase, no spaces/hyphens)
def uniform_format(plates):
    return [plate.replace(' ', '').replace('-', '').upper() for plate in plates]

def check_authorization(check_plates, authorized_df):
    authorized_plates = uniform_format(authorized_df['NumberPlate'].astype(str).tolist())
    check_plates_uniform = uniform_format(check_plates)
    results = []
    for plate in check_plates_uniform:
        results.append("Authorized" if plate in authorized_plates else "Unauthorized")
    return results

def ocr(img_np):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(img_np)
    extracted_texts = []
    for (_, text, _) in result:
        extracted_texts.append(text)
    return extracted_texts

st.title('Vehicle Number Plate OCR & Authorization')

# Secure API key from secrets
api_key = st.secrets["ROBOFLOW_API_KEY"]
MODEL_ID = "vehicle-registration-plates-trudk/2"  # Fixed

# Upload Excel or CSV with authorized number plates
auth_file = st.sidebar.file_uploader("Upload Excel or CSV file of authorized plates (column name: 'NumberPlate')", type=['xlsx', 'csv'])

uploaded_file = st.file_uploader("Upload vehicle image", type=['jpg', 'jpeg', 'png'])

# Storage for OCR result
extracted_plates = []

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
        # Take top detection
        prediction = result['predictions'][0]
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        x1, y1 = int(x - width/2), int(y - height/2)
        x2, y2 = int(x + width/2), int(y + height/2)
        cropped_img = img_np[y1:y2, x1:x2]

        display_img = img_np.copy()
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0,255,0), 2)
        st.image(display_img[:, :, ::-1], caption='Detected Number Plate', use_container_width=True)

        # OCR on cropped number plate
        extracted_plates = ocr(cropped_img)
        st.session_state['extracted_plates'] = extracted_plates

        if extracted_plates:
            st.success(f"OCR Detected Plate(s): {', '.join(extracted_plates)}")
            st.image(cropped_img[:, :, ::-1], caption='Cropped Number Plate', use_container_width=False)
        else:
            st.warning("No text detected in number plate.")
    else:
        st.warning("No number plate detected in this image.")
else:
    if 'extracted_plates' in st.session_state:
        del st.session_state['extracted_plates']

# Authorization Workflow
if auth_file:
    # Load authorized number plates
    if auth_file.name.endswith('.xlsx'):
        auth_df = pd.read_excel(auth_file)
    else:
        auth_df = pd.read_csv(auth_file)
    
    # Check detected plates if available
    extracted_plates = st.session_state.get('extracted_plates', [])
    if extracted_plates:
        check_result = check_authorization(extracted_plates, auth_df)
        st.subheader('Detected Plates Authorization Status')
        results_df = pd.DataFrame({'Number Plate': extracted_plates, 'Status': check_result})
        st.dataframe(results_df)
        if check_result[0] == "Authorized":
            st.markdown(
                f'<span style="color:green; font-size:2em; font-weight:bold;">AUTHORIZED</span>',
        unsafe_allow_html=True
    )
        else:
            st.markdown(
                f'<span style="color:red; font-size:2em; font-weight:bold;">UNAUTHORIZED</span>',
        unsafe_allow_html=True
    )

    
    # Manual override
    st.markdown("**Not satisfied with the detection? Enter number plates for manual check (one per line):**")
    input_manual = st.text_area("Manual Input Plates", height=120)
    if input_manual:
        manual_plates = [i.strip() for i in input_manual.split('\n') if i.strip()]
        manual_result = check_authorization(manual_plates, auth_df)
        manual_results_df = pd.DataFrame({'Number Plate': manual_plates, 'Status': manual_result})
        st.subheader('Manual Check Results')
        st.dataframe(manual_results_df)
else:
    st.info("Please upload an Excel or CSV file of authorized number plates in the sidebar.")


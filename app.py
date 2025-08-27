import streamlit as st
import json
from PIL import Image
import os
import pandas as pd

# --- 1. Configuration ---
JSON_FILE_PATH = 'v1filt_v2_lr2e-5-ve-proj-llm_ignorepunc.json'
CSV_FILE_PATH = 'results.csv'  # Path to your new CSV file
IMAGE_BASE_DIRECTORY = '.'

# --- 2. Language Mappings ---
LANG_CODE_TO_FOLDER = {
    'bn': 'Bengali',
    'en': 'English',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'or': 'Odia',
    'pa': 'Punjabi',
    'ta': 'Tamil',
    'te': 'Telugu'
}

FOLDER_TO_LANG_CODE = {v: k for k, v in LANG_CODE_TO_FOLDER.items()}
AVAILABLE_LANGUAGES = sorted(list(LANG_CODE_TO_FOLDER.values()))

# --- 3. Data Loading Function ---
@st.cache_data
def load_data(json_path, csv_path):
    """
    Loads data from both JSON and CSV files, then merges them.
    """
    # Load the primary data from the JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        st.error(f"❌ Error: JSON file not found at '{json_path}'.")
        return None

    # Load the modification data from the CSV and create a mapping
    try:
        csv_df = pd.read_csv(csv_path)
        # Create a dictionary for quick lookup: image_name -> corr_mod
        corr_mod_map = pd.Series(csv_df.corr_mod.values, index=csv_df.image_name).to_dict()
    except FileNotFoundError:
        st.warning(f"⚠️ Warning: CSV file not found at '{csv_path}'. 'corr_mod' data will not be available.")
        corr_mod_map = {} # Use an empty map if the file is missing

    # Organize data by language and inject the corr_mod details
    data_by_lang = {}
    for item in json_data:
        try:
            lang_code = item['image_name'].split('_')[1]
            # Add the corr_mod data to each item using the map
            item['corr_mod'] = corr_mod_map.get(item['image_name'], 'N/A')
            
            if lang_code not in data_by_lang:
                data_by_lang[lang_code] = []
            data_by_lang[lang_code].append(item)
        except IndexError:
            st.warning(f"Skipping item with unexpected image_name format: {item.get('image_name', 'N/A')}")
            
    return data_by_lang

# --- 4. Main Application UI ---
def main():
    st.set_page_config(layout="wide", page_title="OCR Visualiser")
    st.title("📄 OCR Evaluation Visualiser")

    # Load the combined data
    all_data = load_data(JSON_FILE_PATH, CSV_FILE_PATH)
    if not all_data:
        return

    st.sidebar.header("⚙️ Controls")
    selected_language_folder = st.sidebar.selectbox(
        "Choose a language:",
        options=AVAILABLE_LANGUAGES
    )

    selected_lang_code = FOLDER_TO_LANG_CODE[selected_language_folder]

    if selected_lang_code in all_data:
        language_specific_data = all_data[selected_lang_code]
        image_names = [item['image_name'] for item in language_specific_data]

        selected_image_name = st.selectbox(
            f"Select an image snippet from '{selected_language_folder}' ({len(image_names)} available):",
            options=image_names
        )

        if selected_image_name:
            selected_record = next((item for item in language_specific_data if item['image_name'] == selected_image_name), None)

            if selected_record:
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Image Snippet")
                    image_filename = f"{selected_image_name}.png"
                    image_path = os.path.join(IMAGE_BASE_DIRECTORY, selected_language_folder, image_filename)
                    try:
                        image = Image.open(image_path)
                        st.image(image, caption=image_filename, use_container_width=True)
                    except FileNotFoundError:
                        st.error(f"Image not found at: {image_path}")
                
                with col2:
                    st.subheader("Ground Truth (`gt`)")
                    st.text_area("gt", value=selected_record.get('gt', ''), height=150, key="gt_text", label_visibility="collapsed")

                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("Prediction (`pred`)")
                    st.text_area("pred", value=selected_record.get('pred', ''), height=150, key="pred_text", label_visibility="collapsed")
                
                with col4:
                    st.subheader("Corrected (`corr`)")
                    st.text_area("corr", value=selected_record.get('corr', ''), height=150, key="corr_text", label_visibility="collapsed")
                    
                    # ✅ NEW: Display the corr_mod data
                    st.markdown("###### Modification Details")
                    st.code(selected_record.get('corr_mod', 'N/A'), language=None)

    else:
        st.warning(f"No data found for the language code: '{selected_lang_code}'.")

if __name__ == "__main__":
    main()
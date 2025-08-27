import streamlit as st
import json
from PIL import Image
import os

# --- 1. Configuration ---

# ❌ OLD PATHS (WILL NOT WORK)
# JSON_FILE_PATH = '/Users/oikantik/eval-set/v1filt_v2_lr2e-5-ve-proj-llm_ignorepunc.json'
# IMAGE_BASE_DIRECTORY = '/Users/oikantik/eval-set/'

# ✅ NEW RELATIVE PATHS (CORRECT FOR DEPLOYMENT)
JSON_FILE_PATH = 'v1filt_v2_lr2e-5-ve-proj-llm_ignorepunc.json' 
IMAGE_BASE_DIRECTORY = '.' 

# # --- 1. Configuration ---
# # IMPORTANT: Update these paths to match your local file locations.
# JSON_FILE_PATH = '/Users/oikantik/eval-set/v1filt_v2_lr2e-5-ve-proj-llm_ignorepunc.json'
# IMAGE_BASE_DIRECTORY = '/Users/oikantik/eval-set/'  # This is the root directory containing the language folders.

# --- 2. Language Mappings ---
# This dictionary maps the language short codes (from the JSON) to the folder names (from your image).
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

# Invert the mapping to easily find the language code from a selected folder name.
FOLDER_TO_LANG_CODE = {v: k for k, v in LANG_CODE_TO_FOLDER.items()}
AVAILABLE_LANGUAGES = sorted(list(LANG_CODE_TO_FOLDER.values()))

# --- 3. Data Loading Function ---
@st.cache_data  # Caches the data to avoid reloading on every interaction.
def load_data(json_path):
    """
    Loads the JSON data from the specified path.
    Returns a dictionary of data organized by language code, or None if an error occurs.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Organize data into a dictionary for quick filtering.
        data_by_lang = {}
        for item in data:
            try:
                # The language code is the second part of the 'image_name' (e.g., 'br_te_000489_0_0' -> 'te').
                lang_code = item['image_name'].split('_')[1]
                if lang_code not in data_by_lang:
                    data_by_lang[lang_code] = []
                data_by_lang[lang_code].append(item)
            except IndexError:
                # Handle cases where image_name format might be unexpected.
                st.warning(f"Skipping item with unexpected image_name format: {item.get('image_name', 'N/A')}")
        return data_by_lang
    except FileNotFoundError:
        st.error(f"❌ Error: The JSON file was not found at '{json_path}'.")
        st.info("Please update the `JSON_FILE_PATH` variable at the top of the script.")
        return None
    except json.JSONDecodeError:
        st.error(f"❌ Error: Could not decode the JSON file. Please ensure it's a valid JSON.")
        return None

# --- 4. Main Application UI ---
def main():
    """
    The main function that defines the Streamlit app's layout and logic.
    """
    st.set_page_config(layout="wide", page_title="OCR Visualiser")
    st.title("📄 OCR Evaluation Visualiser")

    all_data = load_data(JSON_FILE_PATH)
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
                
                # --- Top Row: Image and Ground Truth ---
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Image Snippet")
                    image_filename = f"{selected_image_name}.png"
                    image_path = os.path.join(IMAGE_BASE_DIRECTORY, selected_language_folder, image_filename)
                    try:
                        image = Image.open(image_path)
                        # MODIFICATION: Reverted to use_column_width to make it fit the grid properly.
                        st.image(image, caption=image_filename, use_column_width='always')
                    except FileNotFoundError:
                        st.error(f"Image not found at: {image_path}")
                        st.warning("Please verify paths and ensure the image file exists.")
                
                with col2:
                    st.subheader("Ground Truth (`gt`)")
                    st.text_area("gt", value=selected_record.get('gt', ''), height=150, key="gt_text", label_visibility="collapsed")

                # --- Bottom Row: Prediction and Corrected ---
                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("Prediction (`pred`)")
                    st.text_area("pred", value=selected_record.get('pred', ''), height=150, key="pred_text", label_visibility="collapsed")
                
                with col4:
                    st.subheader("Corrected (`corr`)")
                    st.text_area("corr", value=selected_record.get('corr', ''), height=150, key="corr_text", label_visibility="collapsed")
    else:
        st.warning(f"No data found in the JSON file for the language code: '{selected_lang_code}'.")

# --- 5. Run the App ---
if __name__ == "__main__":
    main()
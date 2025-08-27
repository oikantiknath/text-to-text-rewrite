import streamlit as st
import json
from PIL import Image
import os
import pandas as pd
import difflib
import html

# --- 1. Configuration ---
JSON_FILE_PATH = 'v1filt_v2_lr2e-5-ve-proj-llm_ignorepunc.json'
CSV_FILE_PATH = 'results.csv'
IMAGE_BASE_DIRECTORY = '.'

# --- 2. Language Mappings ---
LANG_CODE_TO_FOLDER = {
    'bn': 'Bengali', 'en': 'English', 'gu': 'Gujarati', 'hi': 'Hindi',
    'kn': 'Kannada', 'ml': 'Malayalam', 'mr': 'Marathi', 'or': 'Odia',
    'pa': 'Punjabi', 'ta': 'Tamil', 'te': 'Telugu'
}
FOLDER_TO_LANG_CODE = {v: k for k, v in LANG_CODE_TO_FOLDER.items()}
AVAILABLE_LANGUAGES = sorted(list(LANG_CODE_TO_FOLDER.values()))

# --- 3. Helper Function for Diff ---
def create_highlighted_diff(text1, text2):
    """
    Generates an HTML string for text2, highlighting parts that were
    added or changed from text1.
    """
    sm = difflib.SequenceMatcher(None, text1, text2)
    output_html = []
    for opcode, i1, i2, j1, j2 in sm.get_opcodes():
        # Escape text to prevent rendering issues with characters like '<' or '>'
        safe_text_chunk = html.escape(text2[j1:j2])
        if opcode == 'equal':
            output_html.append(safe_text_chunk)
        elif opcode in ('insert', 'replace'):
            output_html.append(f"<mark>{safe_text_chunk}</mark>")
    return "".join(output_html)

# --- 4. Data Loading Function ---
@st.cache_data
def load_data(json_path, csv_path):
    """Loads and merges data from JSON and CSV files."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        st.error(f"❌ Error: JSON file not found at '{json_path}'.")
        return None

    try:
        csv_df = pd.read_csv(csv_path)
        corr_mod_map = pd.Series(csv_df.corr_mod.values, index=csv_df.image_name).to_dict()
    except FileNotFoundError:
        st.warning(f"⚠️ Warning: CSV file not found at '{csv_path}'. 'corr_mod' data will not be available.")
        corr_mod_map = {}

    data_by_lang = {}
    for item in json_data:
        try:
            lang_code = item['image_name'].split('_')[1]
            item['corr_mod'] = corr_mod_map.get(item['image_name'], 'N/A')
            if lang_code not in data_by_lang:
                data_by_lang[lang_code] = []
            data_by_lang[lang_code].append(item)
        except IndexError:
            st.warning(f"Skipping item with unexpected image_name format: {item.get('image_name', 'N/A')}")
    return data_by_lang

# --- 5. Main Application UI ---
def main():
    st.set_page_config(layout="wide", page_title="OCR Visualiser")
    st.title("OCR Evaluation Visualiser")

    # Custom CSS for text blocks and highlighting
    st.markdown("""
        <style>
        .text-block {
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 15px;
            background-color: #f6f8fa;
            height: 200px;
            overflow-y: auto;
            font-family: sans-serif;
            font-size: 16px;
            line-height: 1.6;
        }
        mark {
            background-color: #fff8c5;
            padding: 0.1em 0.2em;
            border-radius: 3px;
        }
        </style>
        """, unsafe_allow_html=True)

    all_data = load_data(JSON_FILE_PATH, CSV_FILE_PATH)
    if not all_data:
        return

    st.sidebar.header("⚙️ Controls")
    selected_language_folder = st.sidebar.selectbox("Choose a language:", options=AVAILABLE_LANGUAGES)
    selected_lang_code = FOLDER_TO_LANG_CODE[selected_language_folder]

    if selected_lang_code in all_data:
        language_specific_data = all_data[selected_lang_code]
        image_names = [item['image_name'] for item in language_specific_data]
        selected_image_name = st.selectbox(f"Select an image snippet from '{selected_language_folder}' ({len(image_names)} available):", options=image_names)

        if selected_image_name:
            selected_record = next((item for item in language_specific_data if item['image_name'] == selected_image_name), None)

            if selected_record:
                st.markdown("---")
                gt_text = html.escape(selected_record.get('gt', ''))
                pred_text = html.escape(selected_record.get('pred', ''))
                corr_text = selected_record.get('corr', '')
                
                # Generate the HTML diff for the corrected text
                diff_html = create_highlighted_diff(selected_record.get('pred', ''), corr_text)
                
                # --- UI Grid ---
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Image Snippet")
                    image_path = os.path.join(IMAGE_BASE_DIRECTORY, selected_language_folder, f"{selected_image_name}.png")
                    try:
                        image = Image.open(image_path)
                        st.image(image, caption=f"{selected_image_name}.png", use_container_width=True)
                    except FileNotFoundError:
                        st.error(f"Image not found at: {image_path}")
                
                with col2:
                    st.subheader("Ground Truth (`gt`)")
                    st.markdown(f'<div class="text-block">{gt_text}</div>', unsafe_allow_html=True)

                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("Prediction (`pred`)")
                    st.markdown(f'<div class="text-block">{pred_text}</div>', unsafe_allow_html=True)
                
                with col4:
                    st.subheader("Corrected (with highlights)")
                    st.markdown(f'<div class="text-block">{diff_html}</div>', unsafe_allow_html=True)
                    st.markdown("###### Modification Details")
                    st.code(selected_record.get('corr_mod', 'N/A'), language=None)
    else:
        st.warning(f"No data found for the language code: '{selected_lang_code}'.")

if __name__ == "__main__":
    main()
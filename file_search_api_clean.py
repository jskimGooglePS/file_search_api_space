# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import os
import time
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError
from typing import Any, Tuple, Dict
import tempfile
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-2.5-pro"
# PDF_FILE_PATHS = ["assets/pdf/image_1.pdf", "assets/pdf/image_2.pdf", "assets/pdf/image_3.pdf", "assets/pdf/image_4.pdf"]
EXCEL_FILE_PATHS = ["data/assets/excel/Data_for_Test_v2.xlsx"]
INPUT_EXCEL = "data/input/questions_for_evaluation.xlsx" 
OUTPUT_EXCEL = "data/output/evaluation_results_cus_2.5_2.xlsx"
STORE_DISPLAY_NAME = "Tony_RAG_Evaluation_Store"

# Initialize Client
client = genai.Client(
    api_key= os.environ.get("GOOGLE_API_KEY")
)
def cleanup(store_name: str):
    """Deletes the File Search Store to clean up resources."""
    print(f"\n--- Cleanup: Deleting Store {store_name} ---")
    try:
        client.file_search_stores.delete(name=store_name, config={'force': True})
        print("Store deleted successfully.")
    except Exception as e:
        print(f"Error deleting store: {e}")

def get_store_id_by_name(display_name: str) -> str:
    """
    지정한 display_name과 일치하는 스토어의 고유 ID(name)를 찾아 반환합니다.
    """
    print(f"Searching for store with name: {display_name}...")
    
    # 1. 사용 가능한 모든 스토어 목록 가져오기
    stores = client.file_search_stores.list()
    
    # 2. 루프를 돌며 display_name 비교
    if stores:
        for store in stores:
            if store.display_name == display_name:
                print(f"Found existing store: {store.name}")
                return store.name
    
    # 3. 찾지 못했을 경우
    print(f"No store found with display_name: {display_name}")
    return None

def main():
    """Main execution function for the RAG evaluation workflow."""
    store_name = get_store_id_by_name(STORE_DISPLAY_NAME)
    if store_name:
        cleanup(store_name)
    print(store_name)

if __name__ == "__main__":
    main()
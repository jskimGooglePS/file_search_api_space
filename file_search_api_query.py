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
    # vertexai=True,
    # project="beha-data",
    # location="us-central1"
)

PROMPT_TEMPLATE = """You are an expert agent responsible for retrieving information for user questions using a RAG system.
Understand the context by combining the original user question and the previous question, and then call the appropriate tool(s).
Your mission is to answer user queries precisely using the provided context files (Excel sheets converted to markdown, PDFs, or Text documents).

**GLOBAL INSTRUCTIONS:**

1.  **Analyze Data Format:**
    * **If the context is a TABLE (Markdown/CSV):** - Treat the first few rows as headers. If headers look combined (e.g., "Category - Year - Q1"), understand this represents a hierarchy.
        - Pay attention to row/column intersections to extract exact values.
        - If a cell is empty, infer its value from the context of the row or the column header (it might mean "0", "Not Applicable", or "Same as above" based on surrounding data).
    * **If the context is TEXT (PDF/Docs):** - synthesize information across multiple chunks if necessary.
        - Distinguish between a main heading and body text.

2.  **Reasoning Process (Chain of Thought):**
    - **Step 1: Search.** Identify which file(s) and sections contain the keywords from the user's query.
    - **Step 2: Verification.** Check if the retrieved data explicitly answers the question. Do not guess.
    - **Step 3: Synthesis.** If the answer requires comparing data points (e.g., "Compare 2023 vs 2024"), perform the mental calculation or side-by-side comparison before generating the text.

3.  **Ambiguity Handling:**
    - If the user asks for a generic term (e.g., "Total Cost") and the file contains specific variants (e.g., "Total Cost (USD)", "Total Cost (EUR)"), explicitly mention which one you are using or ask for clarification if critical.

4.  **Response Guidelines:**
    - **Tone:** Professional, objective, and direct.
    - **Format:** Use Markdown tables for data comparisons. Use bullet points for textual summaries.
    - **Language:** Answer in the same language as the User Query (English, Vietnamese, Korean, etc.).
    - **Citations:** ALWAYS reference the source file name when providing specific facts or numbers (e.g., "According to [report.xlsx]...").

Availble tools are as follows:
{tool_descriptions}

User Query:
{query}
"""

FILE_SEARCH_DESCRIPTION = """
- file_search: A specialized Retrieval-Augemented Generation (RAG) tool. It automatically searchs through the uploaded knowledge base to find relevant context, facts, informations to answer the user's query. It provides grounded answers with citations.
"""

def extract_metadata(response: types.GenerateContentResponse) -> Dict[str, Any]:
    """Extracts Grounding Metadata and summarizes Citations from the response."""
    
    result = {
        'Answer': response.text,
        'Citations_Summary': "No Citation Found",
        'Usage_Metadata': str(response.usage_metadata),
    }
    print(f"Metadata: {result}")
    
    candidate = response.candidates[0] if response.candidates else None
    
    if candidate and candidate.grounding_metadata:
        gm = candidate.grounding_metadata
        citations = []
        
        # Extract Citations (file names)
        if gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                # print(f"Chunk: {chunk}")
                if hasattr(chunk, 'retrieved_context') and chunk.retrieved_context:
                    context = chunk.retrieved_context

                    if context.title:
                        citations.append(context.title)
            
                    print(f"Citations: {citations}")
            if citations:
                result['Citations_Summary'] = "; ".join(set(citations))

    return result


def run_evaluation_loop(tool: types.Tool, df: pd.DataFrame) -> pd.DataFrame:
    """Iterates through questions, calls the API, and records metrics."""
    
    print(f"\n--- Starting Evaluation Loop ---")
    
    # Configuration to pass the RAG tool
    generate_config = types.GenerateContentConfig(tools=[tool], temperature=0.1)
    
    for index, row in df.iterrows():
        question = row['Question']
        print(f"\nProcessing Q{index + 1}/{len(df)}: {question}")
        
        if pd.isna(question) or question == "":
            df.loc[index, 'Answer'] = "Skipped (No question)"
            continue

        try:
            final_prompt_content = PROMPT_TEMPLATE.format(
                tool_descriptions=FILE_SEARCH_DESCRIPTION,
                query=question
            )
            start_time = time.time()
            
            # Call API
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=final_prompt_content,
                config=generate_config
            )
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            # Extract Data
            metadata = extract_metadata(response)
            
            # Record Results
            df.loc[index, 'Answer'] = response.text
            df.loc[index, 'Time_Response_s'] = round(time_taken, 2)
            df.loc[index, 'Citations_Summary'] = metadata['Citations_Summary']
            df.loc[index, 'Usage_Metadata'] = metadata['Usage_Metadata']
            
            print(f"[SUCCESS] Time: {time_taken:.2f}s")
            
        except APIError as e:
            error_message = f"API Error ({e.status_code}): {e.message}"
            df.loc[index, 'Answer'] = error_message
            print(f"[ERROR] {error_message}")
        except Exception as e:
            df.loc[index, 'Answer'] = f"General Error: {e}"

    return df


# def cleanup(store_name: str):
#     """Deletes the File Search Store to clean up resources."""
#     print(f"\n--- Cleanup: Deleting Store {store_name} ---")
#     try:
#         client.file_search_stores.delete(name=store_name)
#         print("Store deleted successfully.")
#     except Exception as e:
#         print(f"Error deleting store: {e}")

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
    
    # Load questions and prepare DataFrame
    try:
        df_questions = pd.read_excel(INPUT_EXCEL)
        df_questions = df_questions[:2]
        if 'Question' not in df_questions.columns:
            print(f"ERROR: Excel file '{INPUT_EXCEL}' must contain a column named 'Question'.")
            return

        # Initialize result columns
        df_questions['Answer'] = None
        df_questions['Time_Response_s'] = None
        df_questions['Citations_Summary'] = None
        df_questions['Usage_Metadata'] = None
        
    except FileNotFoundError:
        print(f"ERROR: Input file not found: '{INPUT_EXCEL}'. Creating a sample file.")
        pd.DataFrame({"Question": ["What is the main topic of the document?", "What is the key takeaway?"]}).to_excel(INPUT_EXCEL, index=False)
        print("Please fill out the questions in the generated Excel file and run again.")
        return
    except Exception as e:
        print(f"ERROR loading Excel: {e}")
        return
    
    store_name = get_store_id_by_name(STORE_DISPLAY_NAME)
    file_search_tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[store_name]
        )
    )
    # Run Evaluation
    df_results = run_evaluation_loop(file_search_tool, df_questions)
    
    # Save Results and Cleanup
    try:
        df_results.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\n[FINISHED] Evaluation results saved to: {OUTPUT_EXCEL}")
    except Exception as e:
        print(f"ERROR saving Excel: {e}")

if __name__ == "__main__":
    main()
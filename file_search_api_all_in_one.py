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

# PROMPT_TEMPLATE_V0 = """
# You are an expert agent responsible for retrieving information for user questions using a RAG system.
# Understand the context by combining the original user question and the previous question, and then call the appropriate tool(s).

# **Rule**:
# - JUST ANSWER SIMPLE ANSWER, NO MORE EXPLAINATION.
# - ANSWER WITH FULLY ENGLISH AS POSSIBLE.

# Availble tools are as follows:
# {tool_descriptions}

# User Query:
# {query}
# """

# FILE_SEARCH_DESCRIPTION = """
# - file_search: A specialized Retrieval-Augemented Generation (RAG) tool. It automatically searchs through the uploaded knowledge base to find relevant context, facts, informations to answer the user's query. It provides grounded answers with citations.
# """

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

def preprocess_excel_to_markdown(file_path:str) -> str:
    """
    Reads an Excel file, flattens hierarchical headers, forward fills category columns, and returns a Markdown string representation optimized for Gemini
    """
    print(f" -> Pre-processing Excel file: {os.path.basename(file_path)}")
    try:
        xl = pd.ExcelFile(file_path)
        all_sheets_md = []

        for sheet_name in xl.sheet_names:
            df = None

            candidates = [
                [0, 1, 2],
                [0, 1],
                0
            ]
            
            for header_levels in candidates:
                try:
                    temp_df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_levels)

                    if isinstance(temp_df.columns, pd.MultiIndex):
                        flat_cols = [str(c[-1]) for c in temp_df.columns]
                    else:
                        flat_cols = [str(c) for c in temp_df.columns]
                    
                    total_cols = len(flat_cols)
                    unnamed_count = sum(1 for c in flat_cols if "Unnamed" in str(c))

                    if total_cols > 0 and (unnamed_count / total_cols) < 0.6:
                        df = temp_df
                        break
                except Exception:
                    continue
            
            if df is None:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
                #df = df[:5] # 추후 제거 필요
            if isinstance(df.columns, pd.MultiIndex):
                new_columns = []
                for col in df.columns.values:
                    parts = []
                    for c in col:
                        s = str(c).strip()
                        if "Unnamed" not in s and s.lower() != 'nan':
                            parts.append(s.replace('\n', ' '))

                    clean_col = " - ".join(parts) if parts else "Index"
                    new_columns.append(clean_col)
                df.columns = new_columns
            else:
                df.columns = [str(c).replace('\n', '').strip() for c in df.columns]

            if not df.empty and df.shape[1] >= 1:
                df.iloc[:, 0] = df.iloc[:, 0].ffill()

                if df.shape[1] >=2:
                    col2_numeric = pd.to_numeric(df.iloc[:, 1], errors='coerce')

                    non_numeric_ratio = col2_numeric.isna().sum() / len(df)

                    if non_numeric_ratio > 0.5:
                        df.iloc[:, 1] = df.iloc[:, 1].ffill()
            
            md_table = df.to_markdown(index=False, tablefmt="pipe")

            sheet_context = f"--- SOURCE FILE: {os.path.basename(file_path)} | SHEET: {sheet_name} ---\n{md_table}\n\n"
            all_sheets_md.append(sheet_context)
    
        return "".join(all_sheets_md)

    except Exception as e:
        print(f"   [WARNING] Could not preprocess {file_path} with advanced logic. Error: {e}")
        return ""

def setup_file_search_tool(file_paths: str) -> Tuple[types.Tool, str]:
    """
    Creates the File Search Store, uploads the file, waits for indexing,
    and returns the configured Tool and the Store name.
    """
    print(f"--- Starting File Search Setup for: {file_paths} ---")

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ERROR: File not found at path: {path}")

    # Create File Search Store
    print("1. Creating File Search Store...")
    file_search_store = client.file_search_stores.create(
        config={'display_name': STORE_DISPLAY_NAME}
    )
    store_name = file_search_store.name
    print(f"Store ID: {store_name}")

    operations = []
    temp_files = []

    # Upload and Import File (Direct Upload)
    for path in file_paths:
        file_name = os.path.basename(path)

        upload_path = path
        display_name = file_name

        # Check if Excel file -> Convert into text
        if path.lower().endswith(('.xlsx', 'xls')):
            markdown_content = preprocess_excel_to_markdown(path)

            if markdown_content:
                temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8')
                temp_file.write(markdown_content)
                temp_file.close()

                upload_path = temp_file.name
                temp_files.append(upload_path)
                display_name = f"{file_name}_processed.txt"
                print(f"   -> Converted to temp file: {upload_path}")
        
            print(temp_files, markdown_content)
        print(f"2. Uploading and Indexing file...{file_name}")
        try:
            operation = client.file_search_stores.upload_to_file_search_store(
                file=upload_path,
                file_search_store_name=store_name,
                config={'display_name': display_name}
            )
            print(f"Operation ID: {operation}")
            operations.append(operation)
        except Exception as e:
            print(f"   [ERROR] Failed to upload {display_name}: {e}")

    # Wait for processing (Polling)
    print("3. Waiting for processing to complete (Polling)...")
    for i, op in enumerate(operations):
        file_name = os.path.basename(file_paths[i])
        print(f"File name: {file_name}, OP: {op}")

        while not op.done:
            time.sleep(10)
            op = client.operations.get(op)
        
        print(f"File {file_name} indexed successfully (Status: ACTIVE).")
    
    # [NEW] Cleanup temp file
    print("4. Cleaning up temporary files...")
    for temp_path in temp_files:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    # Configure Tool
    file_search_tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[store_name]
        )
    )
    
    return file_search_tool, store_name


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


def cleanup(store_name: str):
    """Deletes the File Search Store to clean up resources."""
    print(f"\n--- Cleanup: Deleting Store {store_name} ---")
    try:
        client.file_search_stores.delete(name=store_name)
        print("Store deleted successfully.")
    except Exception as e:
        print(f"Error deleting store: {e}")


def main():
    """Main execution function for the RAG evaluation workflow."""
    
    # Load questions and prepare DataFrame
    try:
        df_questions = pd.read_excel(INPUT_EXCEL)
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
    
    # Setup RAG System
    store_name = None
    try:
        file_search_tool, store_name = setup_file_search_tool(EXCEL_FILE_PATHS)
    except Exception as e:
        print(f"SETUP FAILED. Exiting. {e}")
        return
    
    # Run Evaluation
    df_results = run_evaluation_loop(file_search_tool, df_questions)
    
    # Save Results and Cleanup
    try:
        df_results.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\n[FINISHED] Evaluation results saved to: {OUTPUT_EXCEL}")
    except Exception as e:
        print(f"ERROR saving Excel: {e}")
        
    finally:
        if store_name:
            cleanup(store_name)


if __name__ == "__main__":
    main()
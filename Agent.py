import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from io import StringIO

from together import Together
from e2b_code_interpreter import Sandbox


warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[Code Interpreter Output]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            results = exec.results
        # Get the code execution results (DataFrames, plots, etc.)
        results = exec.results 

        dataset =e2b_code_interpreter.files.read('/home/user/preprocessed_dataset.csv')
     
        # Convert the CSV content string into a DataFrame
        df = pd.read_csv(StringIO(dataset))
        return results,exec.logs.stdout,df

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""


def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error
    


def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str,dataset) -> Tuple[Optional[List[Any]], str]:
    
    # Read the dataset to get column information
    try:
        df = dataset
        columns_info = ", ".join(df.columns.tolist())
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        columns_info = "Unable to read column information"

    # Updated system prompt to include dataset path and columns information
    system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}'.
                    The dataset contains the following columns: {columns_info}

                    You need to analyze the dataset and answer the user's query with a response with one chunk of running Python code to solve them.
                    IMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    

    with st.spinner('Getting response from Together AI LLM model...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        if python_code:
            code_interpreter_results,text_output = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content,text_output
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_message.content
        


def chat_with_llm2(e2b_code_interpreter: Sandbox, dataset_path: str,dataset) -> Tuple[Optional[List[Any]], str]:
    
    # Read the dataset to get column information
    try:
        df = dataset
        columns_info = ", ".join(df.columns.tolist())
        # Capture metadata as strings
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info_str = buffer.getvalue()
        df_head_str = df.head().to_string()

        df_describe_str = df.describe().to_string()
        df_nulls_str = df.isnull().sum().to_string()

    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        columns_info = "Unable to read column information"

    system_prompt= f"""You're a Python data scientist. You are given a dataset at path '{dataset_path}'.

        ### Metadata Overview:
        - **Missing Values Per Column:**
            {df_nulls_str}

        - Data info: 
            {df_info_str}

        - **Statistical Summary:**
            {df_describe_str}

        - **First Few Rows Preview:**
            {df_head_str}

        You need to analyze the dataset and follow the following instructions by providing one chunk of Python code.

        ### Instructions:
            you need to preprocess this dataset by performing the following tasks:
            0. Handling Outliers.
            1. Handling Missing Values:
            - For numerical columns: Fill missing values with the median.
            - For categorical columns: Fill missing values with the mode.
            2. Apply one-hot encoding to all categorical columns.
            3. Standardize all numerical columns using StandardScaler.
            4. Save the cleaned dataset to 'preprocessed_dataset.csv'.

        IMPORTANT:
        - Always use the dataset path variable '{dataset_path}' when reading the CSV file.
        - Ensure the code is efficient, readable, and provides meaningful insights or visualizations.
        - Format your response clearly and concisely.
        """

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    

    with st.spinner('Getting response from Together AI LLM model...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        if python_code:
            code_interpreter_results,text_output,dataset = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content,text_output,dataset
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_message.content,None,None

def main():
    """Main Streamlit application."""
    st.title("📊 AI Data Visualization Agent")
    st.write("Upload your dataset and ask questions about it!")

    # Initialize session state variables
    #if 'together_api_key' not in st.session_state:
    st.session_state.together_api_key = 'd35381cbaedf482aa1a38cc870fb58ad5c13b36ef02d5dce6da592c57cd23909'
    #if 'e2b_api_key' not in st.session_state:
    st.session_state.e2b_api_key = 'e2b_8df323de200fe359d3efe39f48f15ecb8583607e'
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    with st.sidebar:
        st.header("API Keys and Model Configuration")
        st.session_state.together_api_key = 'd35381cbaedf482aa1a38cc870fb58ad5c13b36ef02d5dce6da592c57cd23909'
        st.sidebar.info("💡 Everyone gets a free $1 credit by Together AI - AI Acceleration Cloud platform")
        st.sidebar.markdown("[Get Together AI API Key](https://api.together.ai/signin)")
        
        st.session_state.e2b_api_key = 'e2b_8df323de200fe359d3efe39f48f15ecb8583607e'
        st.sidebar.markdown("[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)")
        
        # Add model selection dropdown
        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0  # Default to first option
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display dataset with toggle
        df = pd.read_csv(uploaded_file)
        st.write("Dataset:")
        show_full = st.checkbox("Show full dataset")
        if show_full:
            st.dataframe(df)
        else:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())

        #The preProcessing: 
        if st.button("PreProcesse"):
            if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    
                    # Pass dataset_path to chat_with_llm
                    code_results, llm_response,text_output,preprocessed_dataset = chat_with_llm2(code_interpreter,dataset_path,df)
                    
                    # Display LLM's text response
                    st.write("AI Response:")
                    st.write(llm_response)
                    
                    # Display results/visualizations
                    if code_results:
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:  # Check if PNG data is available
                                # Decode the base64-encoded PNG data
                                png_data = base64.b64decode(result.png)
                                
                                # Convert PNG data to an image and display it
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="Generated Visualization", use_container_width=False)
                            elif hasattr(result, 'figure'):  # For matplotlib figures
                                fig = result.figure  # Extract the matplotlib figure
                                st.pyplot(fig)  # Display using st.pyplot
                            elif hasattr(result, 'show'):  # For plotly figures
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)  
                    st.write("Code output") 
                    st.write(text_output)
                    st.write(preprocessed_dataset)
        
        # Query input
        query = st.text_area("What would you like to know about your data?",
                            "Can you compare the average cost for two people between different categories?")
        
        if st.button("Analyze"):
            if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    
                    # Pass dataset_path to chat_with_llm
                    code_results, llm_response,text_output = chat_with_llm(code_interpreter, query, dataset_path,df)
                    
                    # Display LLM's text response
                    st.write("AI Response:")
                    st.write(llm_response)
                    
                    # Display results/visualizations
                    if code_results:
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:  # Check if PNG data is available
                                # Decode the base64-encoded PNG data
                                png_data = base64.b64decode(result.png)
                                
                                # Convert PNG data to an image and display it
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="Generated Visualization", use_container_width=False)
                            elif hasattr(result, 'figure'):  # For matplotlib figures
                                fig = result.figure  # Extract the matplotlib figure
                                st.pyplot(fig)  # Display using st.pyplot
                            elif hasattr(result, 'show'):  # For plotly figures
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)  
                    st.write("Code output") 
                    st.write(text_output) 

if __name__ == "__main__":
    main()
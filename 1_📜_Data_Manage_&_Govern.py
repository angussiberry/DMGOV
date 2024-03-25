import streamlit as st
import openai
import requests
import pandas as pd
import random
from io import StringIO
import cohere

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from annotated_text import annotated_text

from utils.SemanticSimilarityExampleSelector2 import SemanticSimilarityExampleSelector2
from utils.FewShotTemplate2 import FewShotPromptTemplate2
from utils.graph import *
from utils.similarity import *
from utils.functions import *

import io
import time
import os


def local_css(file_name):
    with open(file_name) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


local_css("style/style.css")
logo2 = "images/midas_logo.svg"
# st.sidebar.image(logo2)


def generative_data():
    import streamlit as st
    import pandas as pd
    import io

    # ID,PRF,FN,LN,DOB,EM,ADD,PN
    #

    def analyze_data(dataframe):
        # Data description prompt
        prompt = f"""You are a data specialist;
                        - for a given dataset; you need to examine, understand and analyze the date
                        - you then need to create a summary description of the data product published on the platform
                        - you should suggest three potential use cases of certain columns or the whole table
                        - Finally you should asses if any column could contain PII data and its severity

                        The dataset provided has the following columns:
                        {', '.join(dataframe.columns)}

                        And the first few rows of data look like this:
                        {dataframe.head(5).to_markdown(index=False)}

                        You must return the output in the following numbered format

                        1. Summary then the summary of the data set

                        2. Potential Use cases: then list each potential use cases using "-" for the list instead of numbers 

                        3. Identified PII data: then type out the identified PII columns using "-" for the list instead of numbers

                        DO NOT USE ** ANYWHERE IN THE RESPONSE
                        
                        """

        completions = predict_large_language_model_sample(
            "rational-photon-262214",
            "text-bison@001",
            0.3,
            1000,
            0.8,
            40,
            prompt,
        )

        return completions

    def parse_response(response):
        # Process response
        parsed_response = [
            "",
            "",
            "",
        ]  # Summary, Potential Use cases, Potential PII data

        current_section = None

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("1."):
                current_section = 0
            elif line.startswith("2."):
                current_section = 1
            elif line.startswith("3."):
                current_section = 2
            if current_section is not None:
                parsed_response[current_section] += line + "\n"
        return parsed_response

    response3 = ["", "", ""]  # Initialize response3 with default values

    # NM_SFX,FN,LN,DOD,DOB
    st.image(logo, width=500)
    st.header("Intelligent Data Engine", divider="rainbow")
    st.write(
        "Here is where we demonstrate an LLM's ability to infer information where otherwise disparate or missing. Showing its level of intuition gathered from its large and curated corpus of data it is trained on. When grounded on relevant information using Retreivement Augmented Generation (RAG) one can see how this can be applied to Data Silos across a business."
    )

    input_str = st.text_input(
        "Enter column names (comma separated):", value="ID,PRF,FN,LN,DOB,EM,ADD,PN"
    )
    columns = input_str.split(",")

    create_button = st.button("Create Data Dictionary")

    if create_button:
        st.session_state.data_dictionary_csv = generate_data_dictionary(columns)

    if "data_dictionary_csv" in st.session_state:
        st.subheader("Data Dictionary")
        df = pd.read_csv(io.StringIO(st.session_state.data_dictionary_csv))
        st.table(df)

        create_sample_data_button = st.button("Create Sample Data")

        if create_sample_data_button:
            st.session_state.sample_data_csv = generate_sample_data(
                st.session_state.data_dictionary_csv
            )

        if "sample_data_csv" in st.session_state:
            st.subheader("Sample Data")
            sample_df = pd.read_csv(io.StringIO(st.session_state.sample_data_csv))
            st.table(sample_df)

            analyze_data_button = st.button("Analyze Data")

            if analyze_data_button:
                # Truncate data to top 5 rows
                top_5_sample_data = sample_df.head(5)
                top_5_sample_data_str = top_5_sample_data.to_csv(index=False)
                result = analyze_data(top_5_sample_data)
                response3 = parse_response(result)

                # Generate detailed descriptions via OpenAI API
                bank_name = "ABC"
                data_source = "SourceX"
                st.session_state.descriptions = generate_detailed_descriptions(
                    columns, top_5_sample_data_str, bank_name, data_source
                )

            if "descriptions" in st.session_state:
                with st.expander("Business Metadata"):
                    st.write(st.session_state.descriptions)
                    st.session_state.data_quality_rules = get_data_quality_rules3(
                        st.session_state.descriptions
                    )
                with st.expander("Standardisation Rules"):
                    st.write(st.session_state.data_quality_rules)
                with st.expander("Summary"):
                    # Display identified PII data output here
                    st.write(response3[0])
                with st.expander("Use Cases"):
                    st.write(response3[1])  # Display use case output here
                with st.expander("Identified PII Data"):
                    # Display identified PII data output here
                    st.write(response3[2])

                english_query = st.text_input(
                    "Enter an query (in English):",
                    value="List the rows in age order ascending and only show FN LN",
                )
                create_sql_query_button = st.button("Create SQL Query")

                if create_sql_query_button:
                    sample_data_str = sample_df.head(5)
                    sql_response = generate_sql_query(english_query, sample_data_str)

                    # Display the SQL response
                    st.subheader("Generated SQL Query")
                    st.code(sql_response)


def full_metadata_page():
    st.image(logo, width=500)
    st.header("Data Management Module Workflow", divider="rainbow")

    def dataframe_to_excel(df, filename):
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"

        df.to_excel(filename, index=False)
        print(f"The data has been written to {filename}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # llm=CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    #                model_type="llama",
    #                config={'max_new_tokens':50,
    #                        'temperature':0.01})

    # streamlit run demo2.py --theme.base="dark" --theme.primaryColor="#9a9a9a" --theme.backgroundColor="#1b1d23" --theme.secondaryBackgroundColor="#0b4a0e" --theme.textColor="#ececec"

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )

    # Specify the file path
    file_path = "data/demo_excel.xlsx"

    # Upload the file
    uploaded_file = st.file_uploader("Choose a file")

    def workflow(file):
        df = pd.read_excel(file)

        # Display a preview of the data
        st.write(df.head(100))
        time.sleep(5)  # Add delay

        # Calculate the number of empty rows
        empty_rows = df.isnull().sum(axis=1)
        total_empty_rows = (empty_rows > 0).sum()
        total_rows = df.shape[0]
        # Add delay

        with st.expander("See Metadata Report"):
            st.subheader("Metadata report :page_facing_up:")
            st.write(f"Number of empty rows: {total_empty_rows} / {total_rows}")
            st.write(
                f"Percentage of rows with empty values: {total_empty_rows / total_rows * 100}%"
            )
            st.write(
                f"Completion Percentage: {(total_rows - total_empty_rows) / total_rows * 100}%"
            )

        # time.sleep(5)  # Add delay

        # Split the dataframe into two: one with complete rows and one with empty values
        df_complete = df.dropna()

        examples2 = df_complete.apply(
            lambda row: {
                "input": f"{row['COLUMN_NAME']},{row['TABLE_NAME']},{row['DATA_TYPE']},{row['COLUMN_FULL_NAME']}",
                "output": row["COLUMN_DESCRIPTION"],
            },
            axis=1,
        ).tolist()
        # print(examples2)

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            # This is the list of examples available to select from.
            examples2,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            OpenAIEmbeddings(
                openai_api_key="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"
            ),
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=3,
        )

        similar_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="You are a column description generator, your role is to take a Column Name, Table Name, Data Type and Column full name in the form of an Input. Then you output what a predicted description of the column would be using the Input provided. For example",
            suffix="Using these examples and the following\nInput: {metadata}\nOutput the predicted column description on one line:",
            input_variables=["metadata"],
        )
        df_empty = df[df.isnull().any(axis=1)]

        with st.expander("Retreived Incomplete Metadata Entities"):
            df_empty.head()  # print the table in placeholder

        # time.sleep(1)  # wait for 5 seconds

        df_responses = pd.DataFrame()
        timing1 = 3
        timing2 = 2
        counter = 0

        video_file = open("videos/MIDAS_DM_Metadata_2023.mp4", "rb")
        video_bytes = video_file.read()

        with st.expander("See explanation of the demo process"):
            st.video(video_bytes)

        st.write("To allow time to watch the video after 2mins the workflow will start")
        time.sleep(130)

        placeholder = st.empty()  # save a placeholder
        placeholder.empty()  # remove the table by capturing the placeholder with empty content

        # Call the OpenAI API to complete the missing metadata
        with st.spinner("Generating next column description"):
            with st.expander("Metadata generation"):
                for i, row in df_empty.sample(n=min(100, df_empty.shape[0])).iterrows():
                    counter += 1
                    table_name = row["TABLE_NAME"]
                    column_name = row["COLUMN_NAME"]
                    column_full_name = row["DATA_TYPE"]
                    data_type = row["COLUMN_FULL_NAME"]
                    pre_prompt = (
                        f"{column_name},{table_name},{column_full_name},{data_type}"
                    )
                    time.sleep(timing1)  # Add delay
                    prompt = similar_prompt.format(metadata=pre_prompt)
                    placeholder.text("Retreiving most relevant information...")
                    time.sleep(timing2)
                    placeholder.text(
                        f"Embedding retreival complete and added into prompt\n{prompt}"
                    )
                    response = call_metadata(prompt)
                    # time.sleep(10)
                    result = response.replace("\n", "")
                    print(result)
                    annotated_text(
                        (
                            f"{column_name} from table {table_name} = {result}",
                            "LLM Response",
                            "#83468F",
                        )
                    )
                    total_empty_rows -= 1
                    if counter > 2:
                        timing1 = 0
                        timing2 = 0

        # st.write(df_empty.sample(n=min(20, df_empty.shape[0])))
        with st.expander("See Updated Metadata Report"):
            st.subheader("Updated Metadata report :rainbow_check_mark:")
            st.write(f"Number of empty rows: {total_empty_rows} / {total_rows}")
            st.write(
                f"Percentage of rows with empty values: {total_empty_rows / total_rows * 100}%"
            )
            st.write(
                f"Completion Percentage: {(total_rows - total_empty_rows) / total_rows * 100}%"
            )

        dataframe_to_excel(df_empty, "Generated")
        st.success("Autogenerated Metadata Saved Succesfully")

        # Calculate the number of empty rows
        empty_rows = df.isnull().sum(axis=1)
        total_empty_rows = (empty_rows > 0).sum()
        total_rows = df.shape[0]
        # time.sleep(5)  # Add delay

        # time.sleep(5)  # Add delay

        # Split the dataframe into two: one with complete rows and one with empty values
        df_complete = df.dropna()

        examples2 = df_complete.apply(
            lambda row: {
                "input": f"{row['COLUMN_NAME']},{row['TABLE_NAME']},{row['DATA_TYPE']},{row['COLUMN_FULL_NAME']}",
                "output": row["COLUMN_DESCRIPTION"],
            },
            axis=1,
        ).tolist()
        # print(examples2)

        example_selector = SemanticSimilarityExampleSelector2.from_examples(
            # This is the list of examples available to select from.
            examples2,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            ),
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=5,
        )

        similar_prompt = FewShotPromptTemplate2(
            # We provide an ExampleSelector instead of examples.
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="You are a column description generator, your role is to take a Column Name, Table Name, Data Type and Column full name in the form of an Input. Then you output what a predicted description of the column would be using the Input provided. For example",
            suffix="Using these examples and the following\nInput: {metadata}\nOutput the predicted column description on one line:",
            input_variables=["metadata"],
        )

        df_empty = df[df.isnull().any(axis=1)]
        df_full = df_complete
        df_full["LLAMA2-7b (no training)"] = None
        counter = 0

        placeholder = st.empty()  # save a placeholder
        placeholder.empty()  # remove the table by capturing the placeholder with empty content

        # Call the OpenAI API to complete the missing metadata
        with st.spinner("Generating next column description"):
            with st.expander("Metadata generation"):
                for i, row in df_full.head(5).iterrows():
                    counter += 1
                    table_name = row["TABLE_NAME"]
                    column_name = row["COLUMN_NAME"]
                    column_full_name = row["DATA_TYPE"]
                    data_type = row["COLUMN_FULL_NAME"]
                    column_description = row["COLUMN_DESCRIPTION"]
                    pre_prompt = (
                        f"{column_name},{table_name},{column_full_name},{data_type}"
                    )
                    # time.sleep(timing1)  # Add delay
                    prompt = similar_prompt.format(metadata=pre_prompt)
                    placeholder.text("Retreiving most relevant information...")
                    # time.sleep(timing2)
                    placeholder.text(
                        f"Embedding retreival complete and added into prompt\n{prompt}"
                    )
                    response = call_metadata(prompt)
                    response_df = response.split("Please", 1)[0]
                    df_full.at[i, "LLAMA2-7b (no training)"] = response_df.replace(
                        "\n", ""
                    )
                    result = response_df.replace("\n", "")
                    score = similarity(column_description, result)
                    total_empty_rows -= 1
                    annotated_text(
                        (
                            f"{column_name} from table {table_name} = {result}",
                            f"Score = {score}",
                            "#580a9e",
                        )
                    )
                    if counter > 3:
                        timing1 = 0
                        timing2 = 0

        video_file = open("videos/MIDAS_DM_Metadata_Silos_2023.mp4", "rb")
        video_bytes = video_file.read()

        with st.expander("See how we cover Data Silos"):
            st.video(video_bytes)

        dataframe_to_excel(df_full, "Evaluated")
        st.success("Model Evaluation Saved Succesfully")

    if st.button("Use Demo Data"):
        workflow(file_path)

    if uploaded_file is not None:
        workflow(uploaded_file)


def rule_binding():
    st.image(logo, width=500)
    st.header("Rule Binding Module Workflow", divider="rainbow")

    def dataframe_to_excel(df, filename):
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"

        df.to_excel(filename, index=False)
        print(f"The data has been written to {filename}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # llm=CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    #                model_type="llama",
    #                config={'max_new_tokens':50,
    #                        'temperature':0.01})

    # streamlit run demo2.py --theme.base="dark" --theme.primaryColor="#9a9a9a" --theme.backgroundColor="#1b1d23" --theme.secondaryBackgroundColor="#0b4a0e" --theme.textColor="#ececec"

    def dq_workflow(file):
        df = pd.read_excel(file)

        # Display a preview of the data
        st.write(df.head(100))
        time.sleep(5)  # Add delay

        # Calculate the number of empty rows
        empty_rows = df.isnull().sum(axis=1)
        total_empty_rows = (empty_rows > 0).sum()
        total_rows = df.shape[0]
        # Add delay

        with st.expander("See Metadata Report"):
            st.subheader("Metadata report :page_facing_up:")
            st.write(f"Number of empty rows: {total_empty_rows} / {total_rows}")
            st.write(
                f"Percentage of rows with empty values: {total_empty_rows / total_rows * 100}%"
            )
            st.write(
                f"Completion Percentage: {(total_rows - total_empty_rows) / total_rows * 100}%"
            )

        # time.sleep(5)  # Add delay

        # Split the dataframe into two: one with complete rows and one with empty values
        df_complete = df.dropna()

        df_responses = pd.DataFrame()
        timing1 = 3
        timing2 = 2
        counter = 0

        video_file2 = open("videos/MIDAS_DM_Data_Quality&PII_2023.mp4", "rb")
        video_bytes2 = video_file2.read()

        with st.expander("How we are covering Data Quality & PII"):
            st.video(video_bytes2)

        st.write("To allow time to watch the video after 1mins the workflow will start")
        time.sleep(65)

        df_empty = df[df.isnull().any(axis=1)]
        df_full = df_complete
        df_full["LLAMA2-7b (no training)"] = None
        counter = 0

        # Create a 2D numpy array
        np_array = dq_rules()
        # Convert numpy array to pandas DataFrame to add column titles
        df = pd.DataFrame(np_array, columns=["Rule Code", "Data Governance Rule"])

        with st.expander("See Data Governance Rules to be bound to columns"):
            # Display DataFrame in Streamlit
            st.dataframe(df, width=1000)

        placeholder = st.empty()  # save a placeholder
        placeholder.empty()  # remove the table by capturing the placeholder with empty content
        color_code = "#580a9e"
        # Call the OpenAI API to complete the missing metadata
        with st.spinner("Generating Rule Bindings"):
            with st.expander("Rule Binding generation"):
                for i, row in df_full.iterrows():
                    counter += 1
                    Table = row["TABLE_NAME"]
                    Col = row["COLUMN_NAME"]
                    dat_type = row["DATA_TYPE"]
                    col_name = row["COLUMN_FULL_NAME"]
                    desc = row["COLUMN_DESCRIPTION"]

                    outputs = get_rules(Col, Table, dat_type, desc).split(",")
                    for out in outputs:
                        annotated_text(
                            (f"{out}", f"{Col} from table {Table}", color_code)
                        )
                    total_empty_rows -= 1
                    color_code = dark_colors()[random.randint(0, 30)]

        st.success("Model Evaluation Saved Successfully")

    # Upload the file
    uploaded_file = st.file_uploader("Choose a file")

    # Specify the file path
    file_path2 = "data/demo_excel.xlsx"

    if st.button("Use Sample Data"):
        dq_workflow(file_path2)

    if uploaded_file is not None:
        dq_workflow(uploaded_file)


# Load logo image
logo = "images/midas_long.svg"

# st.sidebar.title("📜 Data Management & Governance")
page = st.sidebar.radio(
    "Go to",
    ["Intelligent Data Engine", "Metadata Generation & Benchmarking", "Rule Binding"],
)

st.sidebar.image(logo2)
st.sidebar.success("Select a MIDAS asset to view the sub modules")

# Display the selected page
if page == "Metadata Generation & Benchmarking":
    full_metadata_page()
elif page == "Intelligent Data Engine":
    generative_data()
elif page == "Rule Binding":
    rule_binding()

import openai
import requests
import pandas as pd
import numpy as np
from io import StringIO
import os
import re

# Replace with your OpenAI API key
openai.api_key = "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"


def extract_string(input_str):
    match = re.search(r"(.*?)\"\"\"", input_str)
    if match:
        return match.group(1)
    else:
        return None


def identify_elements(code, code_type):
    if code_type == "spark":
        functions = re.findall(r"spark\.(.*?)\(", code)
        variables = re.findall(r"\${(.*?)}", code)
        table_name = re.search(r"from (.*?)\n", code)
        attributes = re.findall(r"\|([\w_]+),", code)
    elif code_type == "scala":
        functions = re.findall(r"(?<=def )\w+(?=\()", code)
        variables = re.findall(r"(?<=val |var |def )(\w+)\s*[:=]", code)
        if "var targetTableName = (TransactrionRetriever.deriveTablename" in code:
            table_name = re.search(
                r'var targetTableName = (TransactrionRetriever.deriveTablename\("[\w\s,]*"\))',
                code,
            )
        elif "from" in code:
            table_name = re.search(
                r'from\("[\w\s,]*"\))',
                code,
            )
        attributes = re.findall(r'(?<=col\(")(\w+)"\)', code)

    table_name = table_name.group(1) if table_name else "N/A"

    if '""".stripMargin)' in table_name:
        table_name = extract_string(table_name)

    return functions[0], ", ".join(variables), table_name, ", ".join(attributes)


def get_answers(prompt):
    api_key = openai.api_key

    # The API endpoint URL
    api_url = "https://api.openai.com/v1/engines/davinci-codex/completions"
    data = {
        "model": "gpt-3.5-turbo",
        #'prompt': prompt,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,  # Adjust this to control the response length
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Make a POST request to the API
    response = requests.post(api_url, json=data, headers=headers)

    # Get the generated response
    response_data = response.json()

    # print(response_data)
    generated_response = response_data["choices"][0]["message"]["content"]

    return generated_response


# save it to dict
def save_to_dict(prompt, response):
    results = []
    results[prompt] = response
    return results


def select_prompt(selected_option, selected_code_option):
    if selected_option == "Test overview":
        prompt = "Can you please take a minute describe testing processes for this code. please DO NOT write any code"
    elif selected_option == "Unit test":
        prompt = (
            "Can you please write unit test for this code in" + selected_code_option
        )
    elif selected_option == "Validation":
        prompt = (
            "please write a validation test for this code with an example in "
            + selected_code_option
        )
    elif selected_option == "Functional test":
        prompt = (
            "please write a functional test for this code in " + selected_code_option
        )
    else:
        processed_text = "Invalid option selected"
    return prompt


# Function to make API call to OpenAI GPT-3.5 Turbo with source data
def get_mapping(target_data, source_df_top_results, api_key):
    # confidential_context = "This conversation is confidential. Do not store or use the input data for any other purpose."

    # prompt = f"Map the target column '{target_data['target_column_name']}' from table '{target_data['target_table_name']}' to a source column. Provide source_table_name, source_column_name, source_column_datatype and source_column_description. Also generate transformation_rules if required in mapping.)\n"
    # prompt = f"{confidential_context}\nMap the target column '{target_data['target_column_name']}' from table '{target_data['target_table_name']}' to a source column. Provide source_table_name, source_column_name, source_column_datatype, source_column_description and transformation_rules)\n"
    # Append source data for context
    prompt = f"Map the target column '{target_data['target_column_name']}' from table '{target_data['target_table_name']}' to a source column. Provide source_table_name, source_column_name, source_column_datatype and transformation_rules)\n"
    prompt += f"\nSource Data:\n{source_df_top_results.to_string(index=False)}\n"

    # st.write(prompt)
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100,
        n=1,
        temperature=0.4,  # Set temperature
        api_key=api_key,
    )
    return response.choices[0].text.strip()


def predict_large_language_model_sample(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=content,
        temperature=temperature,
        max_tokens=max_decode_steps,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(f"Response from Model: {response}")
    response = response.choices[0].text.strip()
    return response


def call_metadata(input):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=input,
        temperature=0.3,
        max_tokens=100,
        top_p=0.8,
        frequency_penalty=0,
        presence_penalty=0,
    )
    text = response.choices[0].text.strip()
    return text


def call_gpt(input):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=input,
        temperature=0.3,
        max_tokens=100,
        top_p=0.8,
        frequency_penalty=0,
        presence_penalty=0,
    )
    text = response.choices[0].text.strip()
    return text


def call_gpt_custom(input, temp, token):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=input,
        temperature=temp,
        max_tokens=token,
        top_p=0.8,
        frequency_penalty=0,
        presence_penalty=0,
    )
    text = response.choices[0].text.strip()
    return text


def call_gpt_long(input):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=input,
        temperature=0.3,
        max_tokens=2000,
        top_p=0.8,
        frequency_penalty=0,
        presence_penalty=0,
    )
    text = response.choices[0].text.strip()
    return text


def dq_rules():
    # Predefined data quality rules
    data_quality_rules = {
        "rulel": "Must not be NULL",
        "rule2": "Must be in format DD/MM/YYYY",
        "rule3": "Must contain numbers only",
        "rule4": "Must contain letters only",
        "rules": "Must be contain an uppercase",
        "rule6": "Must be in lowercase",
        "rule7": "Must be in format HH:MM: SS",
        "rule8": "Must be a valid email address",
        "rule9": "Must be a valid URL",
        "rulel0": "Must be a valid phone number",
        "rulell": "Must be a valid IP address",
        "rule12": "Must be a valid postal code",
        "rulel3": "Must be a valid country code",
        "rulel4": "Must be a valid currency code",
        "rule15": "Must be a valid language code",
        "rulel6": "Must be a valid credit card number",
        "rulel7": "Must be a valid date",
        "rulel8": "Must be a valid time",
        "rulel9": "Must be a valid datetime",
        "rule20": "Must be a valid percentage",
        "rule21": "Must be a valid integer",
        "rule22": "Must be a valid decimal",
        "rule23": "Must be a valid boolean",
        "rule24": "Must be a valid UUID",
        "rule25": "Must be a valid Social Security Number",
        "rule26": "Must be a valid Tax Identification Number",
        "rule27": "Must be a valid International Bank Account Number",
        "rule28": "Must be a valid Swift Code",
    }

    nump = np.array(list(data_quality_rules.items()))

    return nump


def dark_colors():
    colors = [
        "#000000",
        "#100c08",
        "#004242",
        "#253529",
        "#704241",
        "#414a4c",
        "#36454f",
        "#444c38",
        "#454d32",
        "#483c32",
        "#264348",
        "#353839",
        "#354230",
        "#43302e",
        "#32174d",
        "#333333",
        "#343434",
        "#1c2841",
        "#3c341f",
        "#560319",
        "#004953",
        "#3c1414",
        "#232b2b",
        "#480607",
        "#123524",
        "#321414",
        "#1a2421",
        "#000036",
        "#000039",
        "#2c1608",
        "#1a1110",
    ]

    return colors


def get_data_quality_rules2(metadata_description):
    prompt = f"Given the metadata descriptions:'{metadata_description}', assign three appropriate data quality rules for each description in the same format the following list with the most relevant first: {', '.join (data_quality_rules.values ())}. \n\n"
    rules = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        1000,
        0.8,
        40,
        prompt,
    )
    return rules


def get_prompt(columns):
    prompt = f"Create a data dictionary with the headings 'Name', 'Long Name', 'Data Type', 'Description', in csv format for a table with the following columns: "
    for col in columns:
        prompt += f"{col}, "
    return prompt[:-2]


def generate_data_dictionary(columns):
    prompt = get_prompt(columns)
    response = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        3500,
        0.8,
        40,
        prompt,
    )

    csv_str = response
    lines = csv_str.split("\n")
    processed_lines = [",".join(line.split(",")[: len(columns)]) for line in lines]
    return "\n".join(processed_lines)


def generate_detailed_descriptions(columns, sample_data, bank_name, data_source):
    prompt = f"Please provide detailed descriptions for each of the following columns in the context of a bank called {bank_name} with a data source named {data_source} related to mortgages. Here are the top 5 rows of the data provided:\n\n"
    prompt += sample_data
    prompt += "\n\nHere are the detailed descriptions for each column:\n"

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()


def generate_sample_data(csv_str, num_rows=10):
    prompt = f"Using the csv data dictionary provided \n{csv_str}\n, generate {num_rows} rows of sample data for each of the 'Names' in the dictionary and output it in csv format. IT MUST BE IN CSV FORMAT WITHOUT ```"
    response = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        2700,
        0.8,
        40,
        prompt,
    )
    return response


# Function to get data quality rules using GPT-3
def get_data_quality_rules(metadata_description):
    prompt = f"Given the metadata description:'{metadata_description}', assign three appropriate data quality rules in the same format the following list with the most relevant first: {dq_rules()}. \n\n"
    response = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        1000,
        0.8,
        40,
        prompt,
    )

    rules = response
    return rules.split(" ")


# Function to send SAS code to OpenAI API and return Spark code
def convert_sas_to_spark(sas_code):
    prompt = (
        f"Convert the following SAS code to Spark code: \n\n{sas_code}\n\n Spark code:"
    )
    response = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        1000,
        0.8,
        40,
        prompt,
    )
    spark_code = response.choices[0].text.strip()
    return spark_code


# Function to explain the code using OpenAI API
def explain_code(code):
    prompt = f"Explain the following code in simple terms: \n\n{code}\n\nExplanation:"
    response = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        1000,
        0.8,
        40,
        prompt,
    )
    explanation = response.choices[0].text.strip()
    return explanation


def get_data_descriptions(metadata_description):
    prompt = f"Given the column description: '{metadata_description}', generate a two sentence summary of the description of the column \n\n"
    response = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        1000,
        0.8,
        40,
        prompt,
    )
    rules = response.choices[0].text.strip()
    return rules.split(", ")


def get_key_words(metadata_description):
    prompt = f"Given the metadata description:'{metadata_description}', assign 1-5 Key words like 'Customer', 'Sales' or 'Accounts \n\n"
    response = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        1000,
        0.8,
        40,
        prompt,
    )
    rules = response.choices[0].text.strip()
    return rules.split(", ")


def generate_sql_query(english_query, sample_data):
    prompt = f"Please convert the following English query into an SQL query using the provided sample data:\n\n{english_query}\n\nSample data:\n\n{sample_data}\n\nSQL Query: "

    response2 = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        1000,
        0.8,
        40,
        prompt,
    )

    return response2


def get_data_quality_rules2(col_name, tab_name, dat_typ, col_desc):
    prompt = f"You are a data standardisation rule to column assignment bot, your role is to take a Column Name, Table Name, Data Type and Column description in the form of an Input. Take a breath, then output what rule or rules from the following list are directly relevant to this column \n{dq_rules()}.\n\n For the following input Column Name:'{col_name}', Table Name:'{tab_name}', Data Type:'{dat_typ}' and Column description:'{col_desc}', Output the index number (e.g '1,6,7' refering to rule1, rule6, rule7) for any rule or rules applicable to this column. For example \n\n Input: FN, PARK, STRING, 'This column contains the full name of the person attending the park' \n\n Output: '1,4,5' (referring to rule1, rule4, rule5)\n\n For the following \n\n Input:{col_name}','{tab_name}','{dat_typ}','{col_desc}' \n\n Output:"
    # print(prompt)
    rules = call_gpt(prompt)
    return rules


def get_PII(col_name, tab_name, dat_typ, col_desc):
    prompt = f"Given the following information about a database column, please classify the column's data as Personally Identifiable Information (PII) or not. If the data is not PII, please return 'NA'.\n\nColumn Name: {col_name} \nTable Name: {tab_name} \nData Type: {dat_typ} \nColumn Description: {col_desc} \nPlease provide the PII classification for this data."
    pii = call_gpt(prompt)
    return pii


def get_rules(col_name, tab_name, dat_typ, col_desc):
    rule_indices_list = []
    for _ in range(3):
        rules = get_data_quality_rules2(col_name, tab_name, dat_typ, col_desc).split(
            "'"
        )[1]

        if rules.strip() == "":
            print("NA")
            return

        # Check if 'refering to' is in the string
        if "refering to" in rules:
            # Extract the rule numbers before 'refering to' and before '('
            rule_indices = [
                rule.strip()
                for rule in rules.split("refering to")[0].split("(")[0].split(",")
                if rule.strip()
            ]
        else:
            # If 'refering to' is not in the string, just split by comma
            rule_indices = [rule.strip() for rule in rules.split(",") if rule.strip()]

        rule_indices_list.append(rule_indices)

    # Find common rule_indices in all three calls
    common_rule_indices = list(
        set(rule_indices_list[0]).intersection(*rule_indices_list[1:])
    )

    output_string = ""
    for rule_index in common_rule_indices:
        if rule_index.isdigit():
            if output_string != "":
                output_string += ", "
            output_string += f"Rule {rule_index}: {dq_rules()[int(rule_index)-1, 1]}"

    return output_string


def get_data_quality_rules3(metadata_description):
    prompt = f"Given the metadata descriptions:'{metadata_description}', assign three appropriate data quality rules for each description in the same format the following list with the most relevant first: {dq_rules()}. \n\n"
    rules = predict_large_language_model_sample(
        "rational-photon-262214",
        "text-bison@001",
        0.3,
        1000,
        0.8,
        40,
        prompt,
    )
    return rules


def code_description_edh(user_code, system_msg):
    user_msg = f"Provide an in-depth confluence style documentation analysis of the below code: \n\n{user_code}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    assistant_msg = response["choices"][0]["message"]["content"]

    return assistant_msg


def code_documentation_edh(user_code, system_msg):
    user_msg = f"Add appropriate comments and documentation to the below code to increase readability and make the code more understandable: \n\n{user_code}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    assistant_msg = response["choices"][0]["message"]["content"]

    return assistant_msg

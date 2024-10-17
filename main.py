# v2:子sql的生成加入了前后依赖关系，不再是完全独立的
# 修改了2,3,4,5
# root要在get_selected_db_schemas和main两个地方改
import json
import openai
# from chatgpt import init_chatgpt, ask_llm
import time
import copy
from transformers import AutoTokenizer, T5ForSequenceClassification, AutoModelForSequenceClassification, BertForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
import re
import csv
import random
from collections import defaultdict
import os
from src.test_suite.evaluation import test_suite_evaluation
from tqdm import tqdm
import sqlite3
import tiktoken
import random
from http import HTTPStatus
import dashscope
import anthropic

# def chat_gpt(model, prompt):
#     # openai.api_key = "sk-xx"
#     response = openai.ChatCompletion.create(
#         model=model,
#         # prompt=prompt,
#         messages=[{"role": "user", "content": prompt}],
#         stop=[";"],
#         request_timeout=5
#     )
#     response_clean = [choice["message"]["content"]
#                       for choice in response["choices"]]
#     return dict(
#         response=response_clean,
#         **response["usage"]
#     )


# 返回对应数据库的schema信息,字符串格式


def get_db_schemas(all_db_infos, db_name):
    db_schemas = {}

    for db in all_db_infos:
        table_names_original = db["table_names_original"]
        table_names = db["table_names"]
        column_names_original = db["column_names_original"]
        column_names = db["column_names"]
        column_types = db["column_types"]

        db_schemas[db["db_id"]] = {}

        primary_keys, foreign_keys = [], []
        # record primary keys
        for pk_column_idx in db["primary_keys"]:
            pk_table_name_original = table_names_original[column_names_original[pk_column_idx][0]]
            pk_column_name_original = column_names_original[pk_column_idx][1]

            primary_keys.append(
                {
                    "table_name_original": pk_table_name_original.lower(),
                    "column_name_original": pk_column_name_original.lower()
                }
            )

        db_schemas[db["db_id"]]["pk"] = primary_keys

        # record foreign keys
        for source_column_idx, target_column_idx in db["foreign_keys"]:
            fk_source_table_name_original = table_names_original[
                column_names_original[source_column_idx][0]]
            fk_source_column_name_original = column_names_original[source_column_idx][1]

            fk_target_table_name_original = table_names_original[
                column_names_original[target_column_idx][0]]
            fk_target_column_name_original = column_names_original[target_column_idx][1]

            foreign_keys.append(
                {
                    "source_table_name_original": fk_source_table_name_original.lower(),
                    "source_column_name_original": fk_source_column_name_original.lower(),
                    "target_table_name_original": fk_target_table_name_original.lower(),
                    "target_column_name_original": fk_target_column_name_original.lower(),
                }
            )
        db_schemas[db["db_id"]]["fk"] = foreign_keys

        db_schemas[db["db_id"]]["schema_items"] = []
        for idx, table_name_original in enumerate(table_names_original):
            column_names_original_list = []
            column_names_list = []
            column_types_list = []
            for column_idx, (table_idx, column_name_original) in enumerate(column_names_original):
                if idx == table_idx:
                    column_names_original_list.append(
                        column_name_original.lower())
                    column_names_list.append(
                        column_names[column_idx][1].lower())
                    column_types_list.append(column_types[column_idx])

            db_schemas[db["db_id"]]["schema_items"].append({
                "table_name_original": table_name_original.lower(),
                "table_name": table_names[idx].lower(),
                "column_names": column_names_list,
                "column_names_original": column_names_original_list,
                "column_types": column_types_list
            })

    # return db_schemas
    db_schema = db_schemas[db_name]
    db_schema_str = ""
    # 列信息
    for i in db_schema['schema_items']:
        db_schema_str += i['table_name']+'('
        for j in i['column_names']:
            db_schema_str += j + ','
        db_schema_str = db_schema_str[:-1]+")\n"
    # 主键信息
    db_schema_str += "primary_keys("
    for i in db_schema['pk']:
        db_schema_str += i['table_name_original'] + \
            '.'+i['column_name_original'] + ','
    db_schema_str = db_schema_str[:-1]+')\n'
    # 外键信息
    db_schema_str += "foreign_keys("
    for i in db_schema['fk']:
        db_schema_str += i['source_table_name_original']+'.'+i['source_column_name_original'] + \
            '='+i['target_table_name_original']+'.' + \
            i['target_column_name_original'] + ','
    db_schema_str = db_schema_str[:-1]+')\n'

    # db_schema_str += '\n'
    # print(db_schema_str)
    return db_schema_str

# 返回C3方法删减过的schema信息


def get_selected_db_schemas(index, type):
    root_path = "/home3/xianyiran/text2sql/decomposition_and_DAIL/output/5.23_sciencebenckmark/oncomx/"
    if type == 'bridge-from':
        input_path = root_path + 'schema_bridge_from.json'
    elif type == 'bridge-not':
        input_path = root_path + 'schema_bridge_not.json'
    elif type == 'bridge-where':
        input_path = root_path + 'schema_bridge_where.json'
    elif type == 'combination':
        input_path = root_path + 'schema_combination.json'
    with open(input_path, 'r') as f:
        data = json.load(f)
    # print(data[index]['input_sequence'])
    return data[index]['input_sequence']


def get_real_schemas(index):
    with open(INPUT_PATH, 'r') as f:
        schema = json.load(f)
    schema_str = ""
    for key in schema[index]['used_schema']['tables'].keys():
        schema_str += key + '( '
        for column in schema[index]['used_schema']['tables'][key]:
            schema_str += column + ', '
        schema_str = schema_str[:-2] + ")\n"
    for fk in schema[index]['used_schema']['fks']:
        schema_str += fk + '\n'
    return schema_str


def get_c3_schema(index):
    input_path = "/home3/xianyiran/text2sql/decomposition_and_DAIL/output/10.13_experiments/llms/gpt-3.5/0_c3_selected_schema.json"
    with open(input_path, 'r') as f:
        schemas = json.load(f)
    return schemas[index]['input_sequence']


def get_all_schema(index):
    input_path = "/home3/xianyiran/text2sql/decomposition_and_DAIL/output/10.13_experiments/llms/gpt-3.5/0_all_schema.json"
    with open(input_path, 'r') as f:
        schemas = json.load(f)
    return schemas[index]['input_sequence']


# 后处理，比较简单，后续细化
def post_process(sql):
    sql = sql.replace("```sql", "")
    sql = sql.replace("```", "")
    sql = sql.replace("\t", " ")
    sql = " ".join(sql.replace("\n", " ").split())
    return sql

# 1.问题分解


def decomposition_question(input_path, output_path,  model,  prompt_num):
    # data = None  # 待分解的问题
    with open(input_path, 'r') as f:
        data = json.load(f)

    # init_chatgpt(openai_key, "", model)
    output_n = 1  # without self-consistency
    output_list = []
    index = 0

    for one_data in data:
        print('data decomposition index:'+str(index))
        one_data['question_index'] = index
        index += 1
        # prompt = "###I need you to decompose the original complex question into some sub questions, we assume that the decomposition type \
        #     has 'bridge' and 'combination'. The bridge type question often can be decomposed into two sub questions, \
        #         the combination type question often can be decomposed into three sub questions. I need you to output the decomposed \
        #             sub questions and corresponding decomposition type.\n\n "
        # prompt = prompt + "First I will give you some tips and examples and I need you to remember the tips, and do not make same mistakes.\n\
        #     Tip1: For the bridge decomposition type, we can decompose from superlative words, such as most, fewest, highest, etc. \n\
        #     Tip2: For the combination type, the third sub question must include 'tb1' and 'tb2', if not, they can not be categorized into combination.\n\n"  # some tips
        if prompt_num == 0:  # from
            prompt = "## You are a good question decomposer. Give you a question , I need you to decompose the original complex question into two sub questions, \
            and output the decomposed sub questions.\
            The second sub-question must contain 'tb1'.\n\
            ## Here are some examples:\n\n"
            prompt = prompt + "Question: Show the status of the city that has hosted the greatest number of competitions.\n \
                Sub-Question1: Show the status of the city and the number of competitions they host.\n \
                Sub-Question2: Show the status of the city that has hosted the greatest number of competitions in 'tb1'. \n \
                    \
                Question: which course has most number of registered students?\n \
                Sub-Question1: show courses and the number of registered students of each course.\n \
                Sub-Question2: which course has most number of rigistered students in 'tb1'?\n\
                    \
                Question: List the state in the US with the most invoices.\n\
                Sub-Question1: List the states and number of invoices in the US.\n\
                Sub-Question2: List the state in the US with the most invoices in 'tb1'.\n\
                    \
                Question: Return the structure description of the document that has been accessed the fewest number of times.\n\
                Sub-Question1: Return the structure description of the document and the number of times it was acessed .\n\
                Sub-Question2: Return the structure description of the document that has been accessed the fewest number of times in 'tb1'.\n\
                    \
                Question: Find the claim that has the largest total settlement amount. Return the effective date of the claim.\n\
                Sub-Question1: Find the effective date of the claim and the total settlement amount of each claim?\n\
                Sub-Question2: Return the effective date of the claim that has the largest total settlement amount in 'tb1'.\n\
                    \
                Question: What is the id and name of the enzyme with most number of medicines that can interact as 'activator'?\n\
                Sub-Question1: What is the id and name and the number of medicines of the enzyme that can interact as 'activator'?\n\
                Sub-Question2: What is the id and name of the enzyme with most number of medicines in 'tb1'?\n\
                    \
                Question: List the name of the phone model launched in year 2002 and with the highest RAM size.\n\
                Sub-Question1: List the name of the phone model and their RAM size launched in year 2002.\n\
                Sub-Question2: List the name of the phone model with the highest RAM size in 'tb1'.\n\
                    \
                Question: Show the most common apartment type code among apartments with more than 1 bathroom.\n\
                Sub-Question1: Lists the apartment type codes in apartments with more than 1 bathroom and the number of apartments corresponding to each type.\n\
                Sub-Question2: Show the most common apartment type code in 'tb1'.\n\
                    \
                Question: Find the name and id of the team that won the most times in 2008 postseason.\n\
                Sub-Question1: Find the name, id, and number of wins for the 2008 postseason.\n\
                Sub-Question2: Find the name and id of the team that won the most times in 'tb1'.\n\
                    \
                Question: Show the name of track with most number of races.\n\
                Sub-Question1: Show the name of tracks and their number of races.\n\
                Sub-Question2: Show the name of track with most number of races in 'tb1'.\n\n"  # bridge type :sub question 在from clause
        elif prompt_num == 1:  # not
            prompt = "## You are a good question decomposer. Give you a question , I need you to decompose the original complex question into two sub questions, and output the decomposed sub questions.\
            The second sub-question must contain 'tb1'.\n\
            ## Here are some examples:\n\n"
            prompt = prompt + "Question: What is the average bike availability in stations that are not located in Palo Alto?\n\
                Sub-Question1: Which stations located in Palo Alto?\n\
                Sub-Question2: What is the average bike availabiliy in stations which not in 'tb1'?\n\
                    \
                Question: How many customers do not have an account?\n\
                Sub-Question1: What are the ids of customers who has an account?\n\
                Sub-Question2: How many customers that not in 'tb1'?\n\
                    \
                Question: List the first and last name of the students who do not have any food type allergy.\n\
                Sub-Question1: List the students who have food type allergy.\n\
                Sub-Question2: List the first and last name of the students who is not in 'tb1'.\n\
                    \
                Question: Find the average rating star for each movie that are not reviewed by Brittany Harris.\n\
                Sub-Question1: Find the movie that are reviewed by Brittany Harris.\n\
                Sub-Question2: Find the average rating star for each movie that not in 'tb1'.\n\
                    \
                Question: How many enzymes do not have any interactions?\n\
                Sub-Question1: Find the enzymes that have interactions.\n\
                Sub-Question2: How many enzymes that not in 'tb1'.\n\
                    \
                Question: How many schools do not participate in the basketball match?\n\
                Sub-Question1: List the schools that participate in the basketball match.\n\
                Sub-Question2: How many schools that not in 'tb1'?\n\
                    \
                Question: Find the average ram mib size of the chip models that are never used by any phone.\n\
                Sub-Question1: Find the ram mib size of the chip models that are used by any phone.\n\
                Sub-Question2: Find the average ram mib size of the chip models not in 'tb1'.\n\n"  # bridge type not
        elif prompt_num == 2:  # where
            prompt = "## You are a good question decomposer. Give you a question , I need you to decompose the original complex question into two sub questions, and output the decomposed sub questions.\
            The second sub-question must contain 'value1'.\n\
            ## Here are some examples:\n\n"
            prompt = prompt + "Question: On which day and in which zip code was the min dew point lower than any day in zip code 94107?\n\
                Sub-Question1: Find the min dew point in zip code 94107.\n\
                Sub-Question2: On which day and in which zip code the min dew point was less than 'value1'?\n\
                    \
                Question: Return ids of all the products that are supplied by supplier id 2 and are more expensive than the average price of all products.\n\
                Sub-Question1: Return the average price of all products.\n\
                Sub-Question2: Return ids of all the products that are supplied by supplier id 2 and are more expensive than 'value1'.\n\
                    \
                Question: Find the team names of the universities whose enrollments are smaller than the average enrollment size.\n\
                Sub-Question1: Find the average enrollment size.\n\
                Sub-Question2 : Find the team names of the universities whose enrollments are smaller than 'value1'.\n\
                    \
                Question: Show the name, location, open year for all tracks with a seating higher than the average.\n\
                Sub-Question1: Show the average seating of all tracks.\n\
                Sub-Question2: Show the name, location, open year for all tracks with a seating higher than 'value1'.\n\n"  # bridge type where
        elif prompt_num == 3:  # combination
            prompt = "## You are a good question decomposer. Give you a question , I need you to decompose the original complex question into two sub questions, and output the decomposed sub questions.\
            The third sub-question must contain 'tb1' and 'tb2'.\n\
            ## Here are some examples:\n\n"
            prompt = prompt + "Question: List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.\n\
                Sub-Question1: List the states where the secretary of 'Treasury' department was born.\n\
                Sub-Question2: List the states where the secretary of 'Homeland Security' was born.\n\
                Sub-Question3: List the states both in 'tb1' and 'tb2'.\n\
                \
                Question: What are the names and ids of stations that had more than 14 bikes available on average or were installed in December?\n\
                Sub-Question1: What are the names and ids of stations that had more than 14 bikes available on average?\n\
                Sub-Question2: What are the names and ids of stations that were installed in December?\n\
                Sub-Question3: What are the names and ids that both in 'tb1' and 'tb2'?\n\
                \
                Question: What are the names of all stations that have more than 10 bikes available and are not located in San Jose?\n\
                Sub-Question1: What are the names of all stations that have more than 10 bikes available?\n\
                Sub-Question2: What are the names of all stations that are located in San Jose?\n\
                Sub-Question3: What are the names of all stations that in 'tb1' but not in 'tb2'?\n\
                \
                Question: Find the name of tracks which are in Movies playlist but not in music playlist.\n\
                Sub-Question1: Find the name of tracks which are in Movies playlist.\n\
                Sub-Question2: Find the name of tracks which are in music playlist.\n\
                Sub-Question3: Find the name of tracks which are in 'tb1' but not in 'tb2'.\n\
                \
                Question: Show the shipping charge and customer id for customer orders with order status Cancelled or Paid.\n\
                Sub-Question1: Show the shipping charge and customer id for customer orders with order status Cancelled.\n\
                Sub-Question2: Show the shipping charge and customer id for customer orders with order status Paid.\n\
                Sub-Question3: Show the shipping charge and customer id who is in 'tb1' or 'tb2'.\n\
                \
                Question: Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015.\n\
                Sub-Question1: Find the name and location of the stadiums which some concerts happened in the years of 2014.\n\
                Sub-Question2: Find the name and location of the stadiums which some concerts happened in the years of 2015.\n\
                Sub-Question3: Find the name and location of the stadiums who is both in 'tb1' and 'tb2'.\n\
                \
                Question: Find the names of customers who have used both the service \"Close a policy\" and the service \"New policy application\".\n\
                Sub-Question1: Find the names of customers who have the service \"Close a policy\".\n\
                Sub-Question2: Find the names of customers who have the service \"Close a policy\".\n\
                Sub-Question3: Find the names of customers who is in both 'tb1' and 'tb2'.\n\
                    \
                Question: List the medicine name and trade name which can both interact as 'inhibitor' and 'activitor' with enzymes.\n\
                Sub-Question1: List the medicine name and trade name which can interact as 'inhibitor' with enzymes.\n\
                Sub-Question2: List the medicine name and trade name which can interact as 'activitor' with enzymes.\n\
                Sub-Question3: List the medicine name and trade that both in 'tb1' and 'tb2'.\n\
                    \
                Question: Find the pixels of the screen modes that are used by both phones with full accreditation types and phones with Provisional accreditation types.\n\
                Sub-Question1: Find the pixels of the screen modes that are used by phones with full accreditation types.\n\
                Sub-Question2: Find the pixels of the screen modes that are used by phones with Provisional accreditation types.\n\
                Sub-Question3: Find the pixels of the screen modes that both in 'tb1' and 'tb2'.\n\n"  # combination type
        # elif prompt_num == 4:  # baseline
        #     prompt = prompt + "Question: Show the status of the city that has hosted the greatest number of competitions.\n \
        #         Sub-Question1: Show the status of the city and the number of competitions they host.\n \
        #         Sub-Question2: Show the status of the city who has hosted the greatest number of competitions in 'tb1'. \n \
        #         Decomposition-type: bridge\n\
        #         Question: What is the average bike availability in stations that are not located in Palo Alto?\n\
        #         Sub-Question1: Which stations located in Palo Alto?\n\
        #         Sub-Question2: What is the average bike availabiliy in stations which not in 'tb1'?\n\
        #         Decomposition-type: bridge\n\
        #         Question: On which day and in which zip code was the min dew point lower than any day in zip code 94107?\n\
        #         Sub-Question1: Find the min dew point in zip code 94107.\n\
        #         Sub-Question2: On which day and in which zip code the min dew point was less than \"value\"?\n\
        #         Decomposition-type: bridge\n\
        #         Question: List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.\n\
        #         Sub-Question1: List the states where the secretary of 'Treasury' department was born.\n\
        #         Sub-Question2: List the states where the secretary of 'Homeland Security' was born.\n\
        #         Sub-Question3: List the states both in 'tb1' and 'tb2'.\n\
        #         Decomposition-type: combination\n"  # 四种类型都选第一个

        # xyr 分解加schema
        # schema_str = "# Here are the database schemas relevant to the question:\n"
        # for table in one_data['used_schema']['tables'].keys():
        #     schema_str += table + '('
        #     for column in one_data['used_schema']['tables'][table]:
        #         schema_str += column + ', '
        #     schema_str = schema_str[:-2] +')\n'
        # for fk in one_data['used_schema']['fks']:
        #     schema_str += fk + '\n'
        # prompt += schema_str
        #######
        prompt += "# Please output a Sub-Question prefixed with 'Sub-Question' in the exact format of the examples.\n"
        prompt = prompt + "Question: " + \
            one_data['question'] + "\n"  # orignal question
        prompt = prompt.replace('    ', '')
        # print(prompt)
        prompt_list = [prompt]
        n_repeat = 0
        while True:
            try:
                res = chat_gpt(model, prompt)
                break
            except Exception as e:
                n_repeat += 1
                print(
                    f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                time.sleep(1)
                continue
        # try:
        #     res = chat_gpt(model, prompt)
        # except openai.error.InvalidRequestError:
        #     print(
        #         f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
        #     res = ""
        print('result:'+str(res))
        decomposition_information = res['response'][0]
        # one_data.clear()

        if prompt_num == 0:
            decomposition_type = 'bridge-from'
        elif prompt_num == 1:
            decomposition_type = 'bridge-not'
        elif prompt_num == 2:
            decomposition_type = 'bridge-where'
        elif prompt_num == 3:
            decomposition_type = 'combination'

        if decomposition_type == 'combination':
            try:
                subq3 = decomposition_information.split('Sub-Question3:')[1]
                subq3 = subq3.strip()
                decomposition_information = decomposition_information.split(
                    'Sub-Question3:')[0]
                subq2 = decomposition_information.split('Sub-Question2:')[1]
                subq2 = subq2.strip()
                decomposition_information = decomposition_information.split(
                    'Sub-Question2:')[0]
                subq1 = decomposition_information.split('Sub-Question1:')[1]
                subq1 = subq1.strip()
                one_data['interaction_pred'] = []
                one_data['interaction_pred'].append({'question': subq1})
                one_data['interaction_pred'].append({'question': subq2})
                one_data['interaction_pred'].append({'question': subq3})
                one_data['type_pred'] = 'combination'
            except:
                try:
                    subq3 = decomposition_information.split(
                        'Sub-Question 3:')[1]
                    subq3 = subq3.strip()
                    decomposition_information = decomposition_information.split(
                        'Sub-Question 3:')[0]
                    subq2 = decomposition_information.split(
                        'Sub-Question 2:')[1]
                    subq2 = subq2.strip()
                    decomposition_information = decomposition_information.split(
                        'Sub-Question 2:')[0]
                    subq1 = decomposition_information.split(
                        'Sub-Question 1:')[1]
                    subq1 = subq1.strip()
                    one_data['interaction_pred'] = []
                    one_data['interaction_pred'].append({'question': subq1})
                    one_data['interaction_pred'].append({'question': subq2})
                    one_data['interaction_pred'].append({'question': subq3})
                    one_data['type_pred'] = 'combination'
                except:
                    print("error:   "+decomposition_information)
                    one_data['interaction_pred'] = []
                    one_data['type_pred'] = decomposition_type
        elif decomposition_type[:6] == 'bridge':
            try:
                subq2 = decomposition_information.split('Sub-Question2:')[1]
                subq2 = subq2.strip()
                decomposition_information = decomposition_information.split(
                    'Sub-Question2:')[0]
                subq1 = decomposition_information.split('Sub-Question1:')[1]
                subq1 = subq1.strip()
                one_data['interaction_pred'] = []
                one_data['interaction_pred'].append({'question': subq1})
                one_data['interaction_pred'].append({'question': subq2})
                one_data['type_pred'] = decomposition_type
            except:
                try:
                    subq2 = decomposition_information.split(
                        'Sub-Question 2:')[1]
                    subq2 = subq2.strip()
                    decomposition_information = decomposition_information.split(
                        'Sub-Question 2:')[0]
                    subq1 = decomposition_information.split(
                        'Sub-Question 1:')[1]
                    subq1 = subq1.strip()
                    one_data['interaction_pred'] = []
                    one_data['interaction_pred'].append({'question': subq1})
                    one_data['interaction_pred'].append({'question': subq2})
                    one_data['type_pred'] = decomposition_type
                except:
                    print("error:   "+decomposition_information)
                    one_data['interaction_pred'] = []
                    one_data['type_pred'] = decomposition_type
        else:
            print("error: decompostion_type ")
            one_data['interaction_pred'] = []

        output_list.append(res)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=6)
    return output_list

# 2.1.处理data为dail可处理的格式


def process_data_for_dail(decomposition_result, processed_file):
    with open(decomposition_result, 'r') as f:
        datas = json.load(f)
    result = []
    for index, data in enumerate(datas):
        # 根据类型看有几个子问题需要用dail去处理，bridge只需dail生成第一个子问题对应的SQL，combination则生成前两个子问题对应的SQL
        sub_num = 1 if data['type_pred'][:6] == 'bridge' else 2
        for j in range(sub_num):
            tmp = copy.deepcopy(data)
            tmp['ori_question'] = tmp['question']
            del tmp['interaction_pred']
            del tmp['sql']
            del tmp['query_toks']
            # del tmp['query_toks_no_value']
            # tmp['question_index'] = index
            try:
                i = data['interaction_pred'][j]
                tmp['question'] = i['question']
                tmp['question_toks'] = i['question'].split(' ')
            except:
                tmp['question'] = ""
                tmp['question_toks'] = ""
            result.append(tmp)
    with open(processed_file, 'w') as f:
        json.dump(result, f)

# 2.2. 整理最后一个子问题


def process_last_question(input_path, output_path):
    with open(input_path, 'r') as f:
        datas = json.load(f)
    result = []
    for index, data in enumerate(datas):
        tmp = copy.deepcopy(data)
        tmp['ori_question'] = tmp['question']
        del tmp['interaction_pred']
        del tmp['sql']
        # del tmp['question_toks']
        # del tmp['query_toks_no_value']
        # tmp['question_index'] = index
        try:
            i = data['interaction_pred'][-1]
            tmp['question'] = i['question']
            # toks = re.split("([;,!.()?' ])", i['question'])
            # toks = toks.remove("")
            # tmp['question_toks']=toks.remove(" ")
            tmp['question_toks'] = i['question'].split(' ')
        except:
            tmp['question'] = ""
            tmp['question_toks'] = ""
        result.append(tmp)

    with open(output_path, 'w') as f:
        json.dump(result, f)


# 3.1.让dail去生成各个子问题的sql
# 去dail源代码执行

# 3.2.生成含依赖关系的最后一个子SQL
# output为了和dail统一一样是txt
# 例子全部来自extra_train

def last_sql_generate(input_path, output_path):
    model = 'gpt-3.5-turbo'
    # table_path = "/Users/xianyiran/Desktop/实验室/text2sql/spider/tables.json"
    # all_db_infos = json.load(open(table_path))

    with open(input_path, 'r') as f:
        result = json.load(f)

    sqls = []
    for index, i in enumerate(result):
        print(f"question index {index}")
        prompt = "## You need to follow these rules to write an intermediate representation for later SQL generation:\n"
        # prompt += "## tip1: 'tb1','tb2'or 'value1' is a placeholder that you can use directly in the generated SQL.\n\
        #     ## tip2: All the ORDER BY operations are performed by using the placeholder 'counter'.\n"
        if i == {}:
            continue
        elif i['type_pred'] == "bridge-from":
            prompt += "## rule1: 'tb1' is a table that contains all the columns you need. The SELECT operation must look up from 'tb1'.\n\
            ## rule2: All the ORDER BY operations are performed by using the placeholder 'counter'.\n\
            ## rule3: The rest of the syntax for the intermediate representation is the same as SQL.\n\
            # Here are some examples:\n\n"
            prompt += "Question: Show the name of the customer who has the most orders in 'tb1'.\n\
            intermediate representation: SELECT customer_name FROM 'tb1' ORDER BY 'counter' DESC LIMIT 1\n\
            Question: What are the details of the student who registered for the most number of courses in 'tb1'?\n\
            intermediate representation: SELECT student_details FROM 'tb1' ORDER BY 'counter' DESC LIMIT 1 \n\
            Question: What is the full name of the employee who has the most customers in 'tb1'?\n\
            intermediate representation: SELECT first_name , last_name FROM 'tb1' ORDER BY 'counter' DESC LIMIT 1\n\
            Question: What is the name of the document with the most number of sections in 'tb1'?\n\
            intermediate representation: SELECT document_name FROM 'tb1' ORDER BY 'counter' DESC LIMIT 1\n\
            Question: What is the status of the city that has hosted the most competitions in 'tb1'?\n\
            intermediate representation: SELECT Status FROM 'tb1' ORDER BY 'counter' DESC LIMIT 1\n\
            Question: What is the customer first, last name and id with least number of accounts in 'tb1'?\n\
            intermediate representation: SELECT customer_first_name , customer_last_name , customer_id FROM 'tb1' ORDER BY 'counter' ASC LIMIT 1\n\n"
        elif i['type_pred'] == "bridge-not":
            prompt += "## rule1: 'tb1' is a placeholder that you must use after NOT IN operation.\n\
            ## rule2: The rest of the syntax for the intermediate representation is the same as SQL.\n\
            # Here are some examples:\n\n"
            prompt += "Question: How many colors are not in'tb1'?.\n\
            intermediate representation: SELECT count(*) FROM Ref_colors WHERE color_code NOT IN 'tb1'\n\
            Question: Find the number of students who are older than 18 and not in 'tb1'.\n\
            intermediate representation: SELECT count(*) FROM Student WHERE age  >  18 AND StuID NOT IN 'tb1'\n\
            Question: Find the address and staff number of the shops that do not in 'tb1'\n\
            intermediate representation: SELECT address ,  num_of_staff FROM shop WHERE shop_id NOT IN 'tb1'\n\
            Question: How many artists do not in 'tb1'?\n\
            intermediate representation: SELECT count(*) FROM ARTIST WHERE artistid NOT IN 'tb1'\n\
            Question: Find the number of users who not in 'tb1'.\n\
            intermediate representation: SELECT count(*) FROM useracct WHERE u_id NOT IN 'tb1'\n\
            Question: How many drivers are not in 'tb1'?\n\
            intermediate representation: SELECT count(DISTINCT driverId) FROM results WHERE raceId NOT IN 'tb1'\n\n"
        elif i['type_pred'] == "bridge-where":
            prompt += "## rule1: In the condition after the WHERE operator, the value being compared must be replaced with the placeholder 'value1'.\n\
            ## rule2: The rest of the syntax for the intermediate representation is the same as SQL.\n\
            # Here are some examples:\n\n"
            prompt += "Question: Find the id and address of the shops whose score is below 'value1'.\n\
            intermediate representation: SELECT shop_id ,  address FROM shop WHERE score  <  'value1'\n\
            Question: What are the full names and hire dates for employees in the same department as 'value1'?\n\
            intermediate representation: SELECT first_name ,  last_name ,  hire_date FROM employees WHERE department_id  = 'value1'\n\
            Question: Among all the claims, which claims have a claimed amount larger than 'value1'?\n\
            intermediate representation: SELECT Date_Claim_Made ,  Date_Claim_Settled FROM Claims WHERE Amount_Claimed  > 'value1'\n\
            Question: Find the first and last name of students whose age is younger than 'value1'.\n\
            intermediate representation: SELECT fname ,  lname FROM student WHERE age  <  'value1'\n\
            Question: What are the customer id and name corresponding to accounts with a checking balance less than 'value1'?\n\
            intermediate representation: SELECT T1.custid ,  T1.name FROM accounts AS T1 JOIN checking AS T2 ON T1.custid  =  T2.custid WHERE T2.balance  < 'value1'\n\
            Question: Find the distinct driver id and the stop number of all drivers that have a shorter pit stop duration than 'value1'\n\
            intermediate representation: SELECT DISTINCT driverid ,  STOP FROM pitstops WHERE duration  <  'value1'\n\n"
        elif i['type_pred'] == "combination":
            prompt += "## rule1: 'tb1' and 'tb2' are tables that contains all the columns you need. The SELECT operation must look up from 'tb1' or 'tb2'.\n\
            ## rule2: The rest of the syntax for the intermediate representation is the same as SQL.\n\
            # Here are some examples:\n\n"
            prompt += "Question: Find the names of states that both in 'tb1' and 'tb2'.\n\
            intermediate representation: SELECT state FROM 'tb1' INTERSECT SELECT state FROM 'tb2'\n\
            Question: Find the name of product that is both in 'tb1' and 'tb2'.\n\
            intermediate representation: SELECT name FROM 'tb1' INTERSECT SELECT name FROM 'tb2'\n\
            Question: What are the names and ids of stations that in 'tb1' or in 'tb2'?\n\
            intermediate representation: SELECT name ,  id FROM 'tb1' UNION SELECT name ,  id FROM 'tb2'\n\
            Question: Find the name of the campuses that is in 'tb1' or in 'tb2'.\n\
            intermediate representation: SELECT campus FROM 'tb1' UNION SELECT campus FROM 'tb2'\n\
            Question: Find the name of tracks which are in 'tb1' but not in 'tb2'.\n\
            intermediate representation: SELECT name FROM 'tb1' EXCEPT SELECT name FROM 'tb2'\n\
            Question: Which parties have delegates in both 'tb1' and 'tb2'?\n\
            intermediate representation: SELECT Party FROM 'tb1' INTERSECT SELECT Party FROM 'tb2'\n\n"
        prompt += "## Here are the schema information of the database:\n"
        # prompt += get_db_schemas(all_db_infos, i['db_id'])
        # prompt += get_selected_db_schemas(index, i['type_pred'])
        # prompt += get_real_schemas(index)
        # prompt += get_c3_schema(i['question_index']) +'\n'
        prompt += get_c3_schema(index) + '\n'
        prompt += '## Question: ' + i['question'] + "\n"
        prompt += "# Please output the intermediate representation directly in the example format without any explanation\n"
        prompt += "## intermediate representation: "
        prompt = prompt.replace("    ", "")
        # print(prompt)
        n_repeat = 0
        while True:
            try:
                res = chat_gpt(model, prompt)
                break
            except Exception as e:
                n_repeat += 1
                print(
                    f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                time.sleep(1)
                continue
        print('result:'+str(res))
        sql = res['response'][0]
        if sql == "":
            sql = "SELECT"
        if i == {}:
            sql = ""
        # 后处理
        sql = post_process(sql)
        sqls.append(sql)
    with open(output_path, 'w') as f:
        for sql in sqls:
            f.write(sql + '\n')

# 4.1处理dail的生成结果


def process_dail_result(processed_file, dail_result, output_processed_file):
    with open(processed_file, 'r') as f:
        ori_datas = json.load(f)
    with open(dail_result, 'r') as f:
        dail_results = f.readlines()

    print(len(dail_results))
    index_list = []
    results = []
    tmp = {}
    tmp['sub_querys'] = []
    tmp['sub_questions'] = []
    for index, data in enumerate(ori_datas):
        # data['pred_sql']=dail_result[i]
        if data['question_index'] not in index_list:
            # if index not in index_list:
            if (index != 0):
                results.append(tmp)
            # tmp = {}
            tmp = copy.deepcopy(data)
            # now_index = tmp['question_index']
            index_list.append(data['question_index'])
            # index_list.append(index)
            tmp['sub_querys'] = []
            tmp['sub_questions'] = []
            # results.append(tmp)
        # print(index)
        tmp['sub_querys'].append(dail_results[index])
        tmp['sub_questions'].append(data['question'])
    results.append(tmp)
    with open(output_processed_file, 'w') as f:
        json.dump(results, f)

# 4.2 处理生成的最后一个子SQL的生成结果
# 参数分别为4.1，2.2，3.2


def process_last_sql(dail_file, lastsql_info, lastsql_result):
    with open(dail_file, 'r') as f:
        ori_datas = json.load(f)
    with open(lastsql_info, 'r') as f:
        lastsql_info = json.load(f)
    with open(lastsql_result, 'r') as f:
        lastsql_results = f.readlines()

    for i in range(len(lastsql_results)):
        for index_j, j in enumerate(ori_datas):
            if index_j == i:
                j['sub_querys'].append(lastsql_results[i])
                break
    for index_i, i in enumerate(lastsql_info):
        for index_j, j in enumerate(ori_datas):
            if j['question_index'] == i['question_index']:
                # if j['ori_question'] == i['ori_question']:
                j['sub_questions'].append(i['question'])
                break
    with open(dail_file, 'w') as f:
        json.dump(ori_datas, f)

# 5.合成最终的SQL


def composite_final_SQL(processed_dail_result, output_path):
    with open(processed_dail_result, 'r') as f:
        data = json.load(f)
    # table_path = "/Users/xianyiran/Desktop/实验室/text2sql/spider/tables.json"
    # all_db_infos = json.load(open(table_path))
    model = 'gpt-3.5-turbo'  # or gpt-4
    # output_n = 1  # without self-consistency
    output_list = []
    for index, one_data in enumerate(tqdm(data)):
        print('data processing index:'+str(index))
        prompt = ""
        if one_data['type_pred'] == 'bridge-from':
            # give some example 2-3
            prompt = "## You are a great SQL writer. We have sub questions of original question and the sub query of the Sub-Question, \
                we need to generate the final SQL of the original question by using the Sub-Queries. \
                ## Pay attention to the following tips: \
                # 1. 'tb1' is a placeholder which refers to the result of the Sub-Question1, you should replace it in the final query. \n\
                # 2. 'counter' is a placeholder which refers to the result of the function COUNT() in Sub-Question1, you should replace it in the final query. \n\
                # 3. There may be some errors in the Sub-Query, please refer to the schema information to correct when generate the final SQL of the original question.\n\
                # Here are some examples:\n\n \
                \
                Question: Show the status of the city that has hosted the greatest number of competitions.\n \
                Sub-Question1: Show the status of the city and the number of competitions they host.\n \
                Sub-Query1: SELECT T1.Status, COUNT(*) FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID GROUP BY T2.Host_city_ID\n\
                Sub-Question2: Show the status of the city that has hosted the greatest number of competitions in 'tb1'. \n \
                Sub-Query2: SELECT Status FROM 'tb1' ORDER BY 'counter' DESC LIMIT 1\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                city(city id,official name,status,area km 2,population,census ranking)\n\
                farm(farm id,year,total horses,working horses,total cattle,oxen,bulls,cows,pigs,sheep and goats)\n\
                farm competition(competition id,year,theme,host city id,hosts)\n\
                competition record(competition id,farm id,rank)\n\
                primary_keys(city.city_id,farm.farm_id,farm_competition.competition_id,competition_record.competition_id)\n\
                foreign_keys(farm_competition.host_city_id=city.city_id,competition_record.farm_id=farm.farm_id,competition_record.competition_id=farm_competition.competition_id)\n\n\
                Final Query:SELECT T1.Status FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID GROUP BY T2.Host_city_ID ORDER BY COUNT(*) DESC LIMIT 1\n\n\
                    \
                    \
                Question: Show the name of aircraft which fewest people have its certificate.\n \
                Sub-Question1: Show the name of aircraft and number of people who have its certificate.\n \
                Sub-Query1: SELECT T2.name, COUNT(*) FROM Certificate AS T1 JOIN Aircraft AS T2 ON T2.aid  =  T1.aid GROUP BY T1.aid\n \
                Sub-Question2: Show the name of aircraft which fewest people have its certificate in 'tb1'.\n \
                Sub-Query2: SELECT name from 'tb1' ORDER BY 'counter' DESC LIMIT 1 \n \
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                flight(flight number,origin,destination,distance,departure date,arrival date,price,airline id)\n\
                aircraft(airline id,name,distance)\n\
                employee(employee id,name,salary)\n\
                certificate(employee id,airline id)\n\
                primary_keys(flight.flno,aircraft.aid,employee.eid,certificate.eid)\n\
                foreign_keys(flight.aid=aircraft.aid,certificate.aid=aircraft.aid,certificate.eid=employee.eid)\n\n\
                Final Query: SELECT T2.name FROM Certificate AS T1 JOIN Aircraft AS T2 ON T2.aid  =  T1.aid GROUP BY T1.aid ORDER BY count(*) DESC LIMIT 1\n\n\
                    \
                    \
                Question: How many tracks does each genre have and what are the names of the top 5?\n \
                Sub-Question1: How many tracks does each genre have and what are their names?\n \
                Sub-Query1: SELECT T1.name ,  COUNT(*) FROM genres AS T1 JOIN tracks AS T2 ON T2.genre_id  =  T1.id GROUP BY T1.id\n \
                Sub-Question2: How many tracks does each genre have and what are the names of the top 5 in 'tb1'? \n\
                Sub-Query2:  SELECT name , 'counter' FROM 'tb1' ORDER BY 'counter' DESC LIMIT 5\n \
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                employees(id,last name,first name,title,reports to,birth date,hire date,address,city,state,country,postal code,phone,fax,email)\n\
                customers(id,first name,last name,company,address,city,state,country,postal code,phone,fax,email,support rep id)\n\
                genres(id,name)\n\
                tracks(id,name,album id,media type id,genre id,composer,milliseconds,bytes,unit price)\n\
                primary_keys(employees.id,customers.id,genres.id,tracks.id)\n\
                foreign_keys(customers.support_rep_id=employees.id,tracks.genre_id=genres.id)\n\n\
                Final Query: SELECT T1.name ,  COUNT(*) FROM genres AS T1 JOIN tracks AS T2 ON T2.genre_id  =  T1.id GROUP BY T1.id ORDER BY count(*) DESC LIMIT 5\n\n"
            prompt += "# Please output with an 'final Query:' prefix directly without explaination. Let's begin!\n"
            prompt = prompt + "Question: " + \
                one_data['ori_question'] + "\n"  # orignal question
            prompt = prompt + "Sub-Question1: " + \
                one_data['sub_questions'][0] + "\n"
            prompt = prompt + "Sub-Query1: " + \
                one_data['sub_querys'][0] + "\n"
            prompt = prompt + "Sub-Question2:" + \
                one_data['sub_questions'][1] + "\n"
            prompt = prompt + "Sub-Query2: " + \
                one_data['sub_querys'][1] + "\n"
        elif one_data['type_pred'] == 'bridge-not':
            prompt = "## You are a great SQL writer. We have sub questions of original question and the sub query of the Sub-Question, \
                we need to generate the final SQL of the original question by using the Sub-Queries. \
                ## Pay attention to the following tips: \
                # 1. 'tb1' in Sub-Query2 is a placeholder which refers to the result of the Sub-Question1, you should replace it in the final query. \n\
                # 2. There may be some errors in the Sub-Query, please refer to the schema information to correct when generate the final SQL of the original question.\n\
                # Here are some examples:\n\n \
                Question: How many departments are led by heads who are not mentioned?\n\
                Sub-Question1: Show the departments that are led by heads who are mentioned.\n\
                Sub-Query1: SELECT department_id FROM management\n\
                Sub-Question2: How many departments are not in 'tb1'?\n\
                Sub-Query2: SELECT count(*) FROM department WHERE department_id NOT IN 'tb1'\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                department(department id,name,creation,ranking,budget in billions,num employees)\n\
                head(head id,name,born state,age)\n\
                management(department id,head id,temporary acting)\n\
                primary_keys(department.department_id,head.head_id,management.department_id)\n\
                foreign_keys(management.head_id=head.head_id,management.department_id=department.department_id)\n\n\
                Final Query: SELECT count(*) FROM department WHERE department_id NOT IN (SELECT department_id FROM management)\n\n\
                    \
                    \
                Question: What is the mean longitude for all stations that have never had more than 10 bikes available?\n\
                Sub-Question1: Show the atations have had more than 10 bikes available.\n\
                Sub-Query1: SELECT station_id FROM status GROUP BY station_id HAVING max(bikes_available)  >  10\n\
                Sub-Question2: What is the mean longitude for the stations that not in 'tb1'?\n\
                Sub-Query2: SELECT avg(long) FROM station WHERE id NOT IN 'tb1'\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                station(id,name,latitude,longitude,dock count,city,installation date)\n\
                status(station id,bikes available,docks available,time)\n\
                trip(id,duration,start date,start station name,start station id,end date,end station name,end station id,bike id,subscription type,zip code)\n\
                primary_keys(station.id,trip.id)\n\
                foreign_keys(status.station_id=station.id)\n\n\
                Final Query: SELECT avg(long) FROM station WHERE id NOT IN (SELECT station_id FROM status GROUP BY station_id HAVING max(bikes_available)  >  10)\n\n\
                    \
                    \
                Question: What is the full name of each student who is not allergic to any type of food.\n\
                Sub-Question1: What is the student who is allergic to any type of food?\n\
                Sub-Query1: SELECT T1.StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  \"food\"\n\
                Sub-Question2: What is the full name of each student who is not in 'tb1'?\n\
                Sub-Query2: SELECT fname ,  lname FROM Student WHERE StuID NOT IN 'tb1'\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                allergy type(allergy name,allergy type)\n\
                has allergy(stuid,allergy)\n\
                student(stuid,last name,first name,age,sex,major,advisor,city code)\n\
                primary_keys(allergy_type.allergy,student.stuid)\n\
                foreign_keys(has_allergy.allergy=allergy_type.allergy,has_allergy.stuid=student.stuid)\n\n\
                Final Query: SELECT fname ,  lname FROM Student WHERE StuID NOT IN (SELECT T1.StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  \"food\")\n\n"
            prompt += "# Please output with an 'final Query:' prefix directly without explaination. Let's begin!\n"
            prompt = prompt + "Question: " + \
                one_data['ori_question'] + "\n"  # orignal question
            prompt = prompt + "Sub-Question1: " + \
                one_data['sub_questions'][0] + "\n"
            prompt = prompt + "Sub-Query1: " + \
                one_data['sub_querys'][0] + "\n"
            prompt = prompt + "Sub-Question2:" + \
                one_data['sub_questions'][1] + "\n"
            prompt = prompt + "Sub-Query2: " + \
                one_data['sub_querys'][1] + "\n"
        elif one_data['type_pred'] == 'bridge-where':
            prompt = "## You are a great SQL writer. We have sub questions of original question and the sub query of the Sub-Question, \
                we need to generate the final SQL of the original question by using the Sub-Queries. \
                ## Pay attention to the following tips: \
                # 1. 'value1' is a placeholder which refers to the result of the Sub-Question1, you should replace them in the final query. \n\
                # 2. There may be some errors in the Sub-Query, please refer to the schema information to correct when generate the final SQL of the original question.\n\
                # Here are some examples:\n\n\
                Question: On which day and in which zip code was the min dew point lower than any day in zip code 94107?\n\
                Sub-Question1: What was the min dew point in zip code 94107?\n\
                Sub-Query1: SELECT min(min_dew_point_f) FROM weather WHERE zip_code  =  94107\n\
                Sub-Question2: On which day and in which zip code was the min dew point lower than 'value1'?\n\
                Sub-Query2: SELECT date ,  zip_code FROM weather WHERE min_dew_point_f  < 'value1'\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                station(id,name,latitude,longitude,dock count,city,installation date)\n\
                status(station id,bikes available,docks available,time)\n\
                trip(id,duration,start date,start station name,start station id,end date,end station name,end station id,bike id,subscription type,zip code)\n\
                weather(date,max dew point f,mean dew point f,min dew point f,zip code)\n\
                primary_keys(station.id,trip.id)\n\
                foreign_keys(status.station_id=station.id)\n\n\
                Final Query: SELECT date ,  zip_code FROM weather WHERE min_dew_point_f  <  (SELECT min(min_dew_point_f) FROM weather WHERE zip_code  =  94107)\n\n\
                    \
                    \
                Question: What is the name and building of the departments whose budget is more than the average budget?\n\
                Sub-Question1: What is the average budget of the departments?\n\
                Sub-Query1: SELECT avg(budget) FROM department\n\
                Sub-Question2: What is the name and building of the departments whose budget is more than 'value1'?\n\
                Sub-Query2: SELECT dept_name ,  building FROM department WHERE budget > 'value1'\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                department(department name,building,budget)\n\
                teaches(id,course id,section id,semester,year)\n\
                student(id,name,department name,total credits)\n\
                primary_keys(department.dept_name,teaches.id,student.id)\n\
                foreign_keys(student.dept_name=department.dept_name)\n\n\
                Final Query: SELECT dept_name ,  building FROM department WHERE budget  >  (SELECT avg(budget) FROM department)\n\n\
                    \
                    \
                Question: Which customer made the largest amount of claim in a single claim? Return the customer details.\n\
                Sub-Question1: What is the largest amount of claim in a single claim?\n\
                Sub-Query1: SELECT max(amount_claimed) FROM claim_headers\n\
                Sub-Question2: Which customer made the amount of claim equals 'value1' in a single claim? Return the customer details.\n\
                Sub-Query2: SELECT t3.customer_details FROM claim_headers AS t1 JOIN policies AS t2 ON t1.policy_id  =  t2.policy_id JOIN customers AS t3 ON t2.customer_id  =  t3.customer_id WHERE t1.amount_claimed  = 'value1'\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                customers(customer id,customer details)\n\
                staff(staff id,staff details)\n\
                policies(policy id,customer id,policy type code,start date,end date)\n\
                claim headers(claim header id,claim status code,claim type code,policy id,date of claim,date of settlement,amount claimed,amount piad)\n\
                claims documents(claim id,document type code,created by staff id,created date)\n\
                claims processing stages(claim stage id,next claim stage id,claim status name,claim status description)\n\
                claims processing(claim processing id,claim id,claim outcome code,claim stage id,staff id)\n\
                primary_keys(customers.customer_id,staff.staff_id,policies.policy_id,claim_headers.claim_header_id,claims_documents.claim_id,claims_processing_stages.claim_stage_id,claims_processing.claim_processing_id)\n\
                foreign_keys(policies.customer_id=customers.customer_id,claim_headers.policy_id=policies.policy_id,claims_documents.created_by_staff_id=staff.staff_id,claims_documents.claim_id=claim_headers.claim_header_id,claims_processing.staff_id=staff.staff_id,claims_processing.claim_id=claim_headers.claim_header_id)\n\n\
                Final Query: SELECT t3.customer_details FROM claim_headers AS t1 JOIN policies AS t2 ON t1.policy_id  =  t2.policy_id JOIN customers AS t3 ON t2.customer_id  =  t3.customer_id WHERE t1.amount_claimed  =  (SELECT max(amount_claimed) FROM claim_headers)\n\n\ "
            prompt += "# Please output with an 'final Query:' prefix directly without explaination. Let's begin!\n"
            prompt = prompt + "Question: " + \
                one_data['ori_question'] + "\n"  # orignal question
            prompt = prompt + "Sub-Question1: " + \
                one_data['sub_questions'][0] + "\n"
            prompt = prompt + "Sub-Query1: " + \
                one_data['sub_querys'][0] + "\n"
            prompt = prompt + "Sub-Question2:" + \
                one_data['sub_questions'][1] + "\n"
            prompt = prompt + "Sub-Query2: " + \
                one_data['sub_querys'][1] + "\n"
        elif one_data['type_pred'] == 'combination':
            prompt = "## We have sub questions of original question and the sub query of the Sub-Question, \
                we need to generate the final SQL of the original question by using the Sub-Queries. \
                ## Pay attention to the following tips:\n \
                # 1. 'tb1' or 'tb2' is a placeholder. 'tb1' refers to the result of the Sub-Question1, and 'tb2' refers to the result of the Sub-Question2. \n\
                # 2. There may be some errors in the Sub-Query, please refer to the schema information to correct when generate the final SQL of the original question.\n\
                # Here are some examples:\n\n \
                \
                Question: List all of the ids for left-footed players with a height between 180cm and 190cm.\n\
                Sub-Question1: List all of the ids for left-footed players.\n\
                Sub-Query1: SELECT player_api_id FROM Player_Attributes WHERE preferred_foot  =  \"left\"\n\
                Sub-Question2: List all of the ids for players with a height between 180cm and 190cm.\n\
                Sub-Query2: SELECT player_api_id FROM Player WHERE height  >=  180 AND height  <=  190\n\
                Sub-Question3: List all of the ids for players both in 'tb1' and 'tb2'\n\
                Sub-Query3: SELECT player_api_id FROM 'tb1' INTERSECT SELECT player_api_id FROM 'tb2'\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                player attributes(id,player fifa api id,player api id,date,overall rating,potential,preferred foot,attacking work rate,defensive work rate,crossing,finishing,heading accuracy,short passing,volleys,dribbling,curve,free kick accuracy,long passing,ball control,acceleration,sprint speed,agility,reactions,balance,shot power,jumping,stamina,strength,long shots,aggression,interceptions,positioning,vision,penalties,marking,standing tackle,sliding tackle,gk diving,gk handling,gk kicking,gk positioning,gk reflexes)\n\
                sqlite sequence(name,seq)\n\
                player(id,player api id,player name,player fifa api id,birthday,height,weight)\n\
                league(id,country id,name)\n\
                country(id,name)\n\
                team(id,team api id,team fifa api id,team long name,team short name)\n\
                team attributes(id,team fifa api id,team api id,date,buildup play speed,buildup play speed class,buildup play dribbling,buildup play dribbling class,buildup play passing,buildup play passing class,buildup play positioning class,chance creation passing,chance creation passing class,chance creation crossing,chance creation crossing class,chance creation shooting,chance creation shooting class,chance creation positioning class,defence pressure,defence pressure class,defence aggression,defence aggression class,defence team width,defence team width class,defence defender line class)\n\
                primary_keys(player_attributes.id,player.id,league.id,country.id,team.id,team_attributes.id)\n\
                foreign_keys(player_attributes.player_api_id=player.player_api_id,player_attributes.player_fifa_api_id=player.player_fifa_api_id,league.country_id=country.id,team_attributes.team_api_id=team.team_api_id,team_attributes.team_fifa_api_id=team.team_fifa_api_id)\n\n\
                Final Query: SELECT player_api_id FROM Player WHERE height  >=  180 AND height  <=  190 INTERSECT SELECT player_api_id FROM Player_Attributes WHERE preferred_foot  =  \"left\"\n\n\
                    \
                    \
                Question: What campuses are located in Northridge, Los Angeles or in San Francisco, San Francisco?\n\
                Sub-Question1: What campuses are located in Northridge, Los Angeles?\n\
                Sub-Query1: SELECT campus FROM campuses WHERE LOCATION  =  \"Northridge\" AND county  =  \"Los Angeles\"\n\
                Sub-Question2: What campuses are located in San Francisco, San Francisco?\n\
                Sub-Query2: SELECT campus FROM campuses WHERE LOCATION  =  \"San Francisco\" AND county  =  \"San Francisco\"\n\
                Sub-Question3: What campuses are in 'tb1' or 'tb2'?\n\
                Sub-Query3: SELECT campus FROM 'tb1' UNION SELECT campus FROM 'tb2'\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                campuses(id,campus,location,county,year)\n\
                csu fees(campus,year,campus fee)\n\
                degrees(year,campus,degrees)\n\
                discipline enrollments(campus,discipline,year,undergraduate,graduate)\n\
                enrollments(campus,year,totalenrollment ay,fte ay)\n\
                faculty(campus,year,faculty)\n\
                primary_keys(campuses.id,csu_fees.campus,degrees.year,discipline_enrollments.campus,enrollments.campus)\n\
                foreign_keys(csu_fees.campus=campuses.id,degrees.campus=campuses.id,discipline_enrollments.campus=campuses.id,enrollments.campus=campuses.id,faculty.campus=campuses.id)\n\n\
                Final Query: SELECT campus FROM campuses WHERE LOCATION  =  \"Northridge\" AND county  =  \"Los Angeles\" UNION SELECT campus FROM campuses WHERE LOCATION  =  \"San Francisco\" AND county  =  \"San Francisco\"\n\n\
                    \
                    \
                Question: Find courses that ran in Fall 2009 but not in Spring 2010.\n\
                Sub-Question1: Find courses that ran in Fall 2009.\n\
                Sub-Query1: SELECT course_id FROM SECTION WHERE semester  =  'Fall' AND YEAR  =  2009\n\
                Sub-Question2: Find courses that ran in Spring 2010\n\
                Sub-Query2: SELECT course_id FROM SECTION WHERE semester  =  'Spring' AND YEAR  =  2010\n\
                Sub-Question3: Find courses that in 'tb1' but not in 'tb2'\n\
                Sub-Query3: SELECT course_id FROM 'tb1' EXCEPT SELECT course_id FROM 'tb2'\n\
                # Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n\
                classroom(building,room number,capacity)\n\
                course(course id,title,department name,credits)\n\
                section(course id,section id,semester,year,building,room number,time slot id)\n\
                primary_keys(classroom.building,department.dept_name,course.course_id,instructor.id,section.course_id,teaches.id,student.id,takes.id,advisor.s_id,time_slot.time_slot_id,prereq.course_id)\n\
                foreign_keys(section.room_number=classroom.room_number,section.course_id=course.course_id)\n\n\
                Final Query: SELECT course_id FROM SECTION WHERE semester  =  'Fall' AND YEAR  =  2009 EXCEPT SELECT course_id FROM SECTION WHERE semester  =  'Spring' AND YEAR  =  2010\n\n"
            prompt += "# Please output with an 'final Query:' prefix directly without explaination. Let's begin!\n"
            prompt = prompt + "Question: " + \
                one_data['ori_question'] + "\n"  # orignal question
            prompt = prompt + "Sub-Question1: " + \
                one_data['sub_questions'][0] + "\n"
            prompt = prompt + "Sub-Query1: " + \
                one_data['sub_querys'][0] + "\n"
            prompt = prompt + "Sub-Question2: " + \
                one_data['sub_questions'][1] + "\n"
            prompt = prompt + "Sub-Query2: " + \
                one_data['sub_querys'][1] + "\n"
            prompt = prompt + "Sub-Question3: " + \
                one_data['sub_questions'][2] + "\n"
            prompt = prompt + "Sub-Query3: " + \
                one_data['sub_querys'][2] + "\n"
        else:
            print('type error')
            exit(-1)
        n_repeat = 0

        # 无论哪种类型最后都加上CoT与schema信息
        prompt += "# Lets think step by step. \n\
                # First, the sub-SQL is combined to form a complete SQL. Second, with the following schema information, consider if there are any errors in the SQL and fix them if there are.\n"
        # prompt += get_db_schemas(all_db_infos, one_data['db_id'])
        # prompt += get_selected_db_schemas(index, one_data['type_pred'])
        prompt += "database schemas:\n"
        # prompt += get_real_schemas(index)
        # prompt += get_all_schema(one_data['question_index']) + '\n'
        prompt += get_all_schema(index) + '\n'
        prompt = prompt.replace('    ', '')
        # print(prompt)
        while True:
            try:
                res = chat_gpt(model, prompt)
                break
            except Exception as e:
                n_repeat += 1
                print(
                    f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                time.sleep(1)
                continue

        # try:

        #     model = "gpt-3.5-turbo"
        #     res = chat_gpt(model, prompt)
        # except openai.error.InvalidRequestError:
        #     print(
        #         f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
        #     res = ""
        print('result:'+str(res))
        final_sql = res['response'][0]
        try:
            final_sql = final_sql.split('Final Query:')[1]
            final_sql = post_process(final_sql.strip())
        except:
            try:
                final_sql = final_sql.split('final Query:')[1]
                final_sql = post_process(final_sql.strip())
            except:
                try:
                    final_sql = final_sql.split('final query:')[1]
                    final_sql = post_process(final_sql.strip())
                except:
                    print("error:   "+final_sql)
                    final_sql = "SELECT"
        print(final_sql)
        final_sql = post_process(final_sql)
        one_data['final_sql'] = final_sql
        output_list.append(res)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=6)
    return output_list


# 6.处理为可评估格式，使用套件
def process_data_for_eval(final_result, eval_gold, eval_pred):
    with open(final_result, 'r') as f:
        data = json.load(f)
    gold = []
    pred = []
    for i in data:
        gold.append(f"{i['query']}\t{i['db_id']}")
        pred.append(i['final_sql'].replace('\n', ''))
    with open(eval_gold, 'w') as f:
        for i in gold:
            f.write(i)
            f.write('\n')
    f.close()
    with open(eval_pred, 'w') as f:
        for i in pred:
            f.write(i)
            f.write('\n')
    f.close()

# 7.手动把分数加过来处理


def eval_score_process(root_path):
    # bridge_from
    bridge_from_score = [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1,
                         0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
    # bridge_where
    bridge_where_score = [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
                          1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
    # bridge_not
    bridge_not_score = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
    # combination
    combination_score = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
                         1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]

    bridge_from_path = root_path+"5_bridge_from.json"
    bridge_where_path = root_path+"5_bridge_where.json"
    bridge_not_path = root_path+"5_bridge_not.json"
    combination_path = root_path+"5_combination.json"
    ####
    with open(bridge_from_path, 'r') as f:
        data = json.load(f)
    f.close()
    for index, i in enumerate(data):
        i['exec_match'] = bridge_from_score[index]
    with open(bridge_from_path, 'w') as f:
        json.dump(data, f)
    f.close()
    ####
    with open(bridge_where_path, 'r') as f:
        data = json.load(f)
    f.close()
    for index, i in enumerate(data):
        i['exec_match'] = bridge_where_score[index]
    with open(bridge_where_path, 'w') as f:
        json.dump(data, f)
    f.close()
    ####
    with open(bridge_not_path, 'r') as f:
        data = json.load(f)
    f.close()
    for index, i in enumerate(data):
        i['exec_match'] = bridge_not_score[index]
    with open(bridge_not_path, 'w') as f:
        json.dump(data, f)
    f.close()
    ####
    with open(combination_path, 'r') as f:
        data = json.load(f)
    f.close()
    for index, i in enumerate(data):
        i['exec_match'] = combination_score[index]
    with open(combination_path, 'w') as f:
        json.dump(data, f)
    f.close()


# 8.统计四个加起来的正确率
def analysis_acc(root_path):
    # 没分解
    # dail_score = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1,
    #               0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]

    extra_path = "extra_dev.json"
    bridge_from_path = root_path+"5_bridge_from.json"
    bridge_where_path = root_path+"5_bridge_where.json"
    bridge_not_path = root_path+"5_bridge_not.json"
    combination_path = root_path+"5_combination.json"

    with open(extra_path, 'r') as f:
        ori_data = json.load(f)
    f.close()
    ####
    with open(bridge_from_path, 'r') as f:
        bridge_from_data = json.load(f)
    f.close()
    ####
    with open(bridge_where_path, 'r') as f:
        bridge_where_data = json.load(f)
    f.close()
    ####
    with open(bridge_not_path, 'r') as f:
        bridge_not_data = json.load(f)
    f.close()
    ####
    with open(combination_path, 'r') as f:
        combination_data = json.load(f)
    f.close()

    correct = 0

    for i in bridge_from_data:
        if 'diff_exec' not in ori_data[i['question_index']].keys():
            ori_data[i['question_index']]['diff_exec'] = []
        ori_data[i['question_index']]['diff_exec'].append(
            {'bridge_from_exec': i['exec_match']})
    for i in bridge_where_data:
        if 'diff_exec' not in ori_data[i['question_index']].keys():
            ori_data[i['question_index']]['diff_exec'] = []
        ori_data[i['question_index']]['diff_exec'].append(
            {'bridge_where_exec': i['exec_match']})
    for i in bridge_not_data:
        if 'diff_exec' not in ori_data[i['question_index']].keys():
            ori_data[i['question_index']]['diff_exec'] = []
        ori_data[i['question_index']]['diff_exec'].append(
            {'bridge_not_exec': i['exec_match']})
    for i in combination_data:
        if 'diff_exec' not in ori_data[i['question_index']].keys():
            ori_data[i['question_index']]['diff_exec'] = []
        ori_data[i['question_index']]['diff_exec'].append(
            {'combination_exec': i['exec_match']})

    correct = 0
    for index, i in enumerate(ori_data):
        flag = 0
        for j in i['diff_exec']:
            if 'bridge_from_exec' in j.keys() and j['bridge_from_exec'] == 1:
                flag = 1
            elif 'bridge_where_exec' in j.keys() and j['bridge_where_exec'] == 1:
                flag = 1
            elif 'bridge_not_exec' in j.keys() and j['bridge_not_exec'] == 1:
                flag = 1
            elif 'combination_exec' in j.keys() and j['combination_exec'] == 1:
                flag = 1
        correct += flag
    print(correct/len(ori_data))
    print(correct)
    print(len(ori_data))
    output_path = root_path+"8_final.json"
    with open(output_path, 'w') as f:
        json.dump(ori_data, f)
    f.close()

# 8.使用lever的verifier进行gold选择


def lever_verify_acc(root_path):
    extra_path = "extra_dev.json"
    bridge_from_path = root_path+"5_bridge_from.json"
    bridge_where_path = root_path+"5_bridge_where.json"
    bridge_not_path = root_path+"5_bridge_not.json"
    combination_path = root_path+"5_combination.json"
    bridge_from_exec_path = root_path+"bridge_from_exec.txt"
    bridge_where_exec_path = root_path+"bridge_where_exec.txt"
    bridge_not_exec_path = root_path+"bridge_not_exec.txt"
    combination_exec_path = root_path+"combination_exec.txt"

    with open(extra_path, 'r') as f:
        ori_data = json.load(f)
    f.close()
    ####
    with open(bridge_from_path, 'r') as f:
        bridge_from_data = json.load(f)
    f.close()
    ####
    with open(bridge_where_path, 'r') as f:
        bridge_where_data = json.load(f)
    f.close()
    ####
    with open(bridge_not_path, 'r') as f:
        bridge_not_data = json.load(f)
    f.close()
    ####
    with open(combination_path, 'r') as f:
        combination_data = json.load(f)
    f.close()
    ########################################
    with open(bridge_from_exec_path, 'r') as f:
        bridge_from_exec = f.readlines()
    f.close()
    ####
    with open(bridge_where_exec_path, 'r') as f:
        bridge_where_exec = f.readlines()
    f.close()
    ####
    with open(bridge_not_exec_path, 'r') as f:
        bridge_not_exec = f.readlines()
    f.close()
    ####
    with open(combination_exec_path, 'r') as f:
        combination_exec = f.readlines()
    f.close()

    model_name = "model/niansong1996/lever-spider-codex"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    pred_sql = []
    for index, data in enumerate(tqdm(ori_data)):
        # 存放四个的分数
        score = []
        # bridge_from
        input_str = "-- question: "
        input_str += data['question'] + '| '
        input_str += "-- SQL:| "
        input_str += bridge_from_data[index]['final_sql']+"| | "
        input_str += "-- exec result:|/*|"
        # 规则成lever要求格式
        exec_tmp = bridge_from_exec[index].replace("'", "")
        exec_tmp = exec_tmp.replace("), (", " | ")
        exec_tmp = exec_tmp.replace(",", " ")
        if exec_tmp[0] == '[':
            exec_tmp = exec_tmp.replace("[", "")
            exec_tmp = exec_tmp.replace("]", "")
            exec_tmp = exec_tmp.replace("(", "")
            exec_tmp = exec_tmp.replace(")", "")
            exec_tmp = exec_tmp.replace("\n", "")
        input_str += exec_tmp + "|*/"
        # print(input_str)
        input_ids = tokenizer.encode(input_str, return_tensors='pt')
        model_result = model.generate(input_ids=input_ids,
                                      do_sample=False, return_dict_in_generate=True,
                                      output_scores=True, num_beams=1)
        logit = model_result.scores[0][0]
        # yes no对应的logits
        soft = torch.stack((logit[4273], logit[150]), 0)
        soft = torch.softmax(soft, 0)
        score.append(float(soft[0]))
        #################################################################################
        # bridge_not
        input_str = "-- question: "
        input_str += data['question'] + '| '
        input_str += "-- SQL:| "
        input_str += bridge_not_data[index]['final_sql']+"| | "
        input_str += "-- exec result:|/*|"
        # 规则成lever要求格式
        exec_tmp = bridge_not_exec[index].replace("'", "")
        exec_tmp = exec_tmp.replace("), (", " | ")
        exec_tmp = exec_tmp.replace(",", " ")
        if exec_tmp[0] == '[':
            exec_tmp = exec_tmp.replace("[", "")
            exec_tmp = exec_tmp.replace("]", "")
            exec_tmp = exec_tmp.replace("(", "")
            exec_tmp = exec_tmp.replace(")", "")
            exec_tmp = exec_tmp.replace("\n", "")
        input_str += exec_tmp + "|*/"
        # print(input_str)
        input_ids = tokenizer.encode(input_str, return_tensors='pt')
        model_result = model.generate(input_ids=input_ids,
                                      do_sample=False, return_dict_in_generate=True,
                                      output_scores=True, num_beams=1)
        logit = model_result.scores[0][0]
        # yes no对应的logits
        soft = torch.stack((logit[4273], logit[150]), 0)
        soft = torch.softmax(soft, 0)
        score.append(float(soft[0]))
        #################################################################################
        # bridge_where
        input_str = "-- question: "
        input_str += data['question'] + '| '
        input_str += "-- SQL:| "
        input_str += bridge_where_data[index]['final_sql']+"| | "
        input_str += "-- exec result:|/*|"
        # 规则成lever要求格式
        exec_tmp = bridge_where_exec[index].replace("'", "")
        exec_tmp = exec_tmp.replace("), (", " | ")
        exec_tmp = exec_tmp.replace(",", " ")
        if exec_tmp[0] == '[':
            exec_tmp = exec_tmp.replace("[", "")
            exec_tmp = exec_tmp.replace("]", "")
            exec_tmp = exec_tmp.replace("(", "")
            exec_tmp = exec_tmp.replace(")", "")
            exec_tmp = exec_tmp.replace("\n", "")
        input_str += exec_tmp + "|*/"
        # print(input_str)
        input_ids = tokenizer.encode(input_str, return_tensors='pt')
        model_result = model.generate(input_ids=input_ids,
                                      do_sample=False, return_dict_in_generate=True,
                                      output_scores=True, num_beams=1)
        logit = model_result.scores[0][0]
        # yes no对应的logits
        soft = torch.stack((logit[4273], logit[150]), 0)
        soft = torch.softmax(soft, 0)
        score.append(float(soft[0]))
        #################################################################################
        # combination
        input_str = "-- question: "
        input_str += data['question'] + '| '
        input_str += "-- SQL:| "
        input_str += combination_data[index]['final_sql']+"| | "
        input_str += "-- exec result:|/*|"
        # 规则成lever要求格式
        exec_tmp = combination_exec[index].replace("'", "")
        exec_tmp = exec_tmp.replace("), (", " | ")
        exec_tmp = exec_tmp.replace(",", " ")
        if exec_tmp[0] == '[':
            exec_tmp = exec_tmp.replace("[", "")
            exec_tmp = exec_tmp.replace("]", "")
            exec_tmp = exec_tmp.replace("(", "")
            exec_tmp = exec_tmp.replace(")", "")
            exec_tmp = exec_tmp.replace("\n", "")
        input_str += exec_tmp + "|*/"
        # print(input_str)
        input_ids = tokenizer.encode(input_str, return_tensors='pt')
        model_result = model.generate(input_ids=input_ids,
                                      do_sample=False, return_dict_in_generate=True,
                                      output_scores=True, num_beams=1)
        logit = model_result.scores[0][0]
        # yes no对应的logits
        soft = torch.stack((logit[4273], logit[150]), 0)
        soft = torch.softmax(soft, 0)
        score.append(float(soft[0]))
        #################################################################################
        max_index = np.argmax(score)
        if max_index == 0:
            pred_sql.append(bridge_from_data[index]['final_sql'])
        elif max_index == 1:
            pred_sql.append(bridge_not_data[index]['final_sql'])
        elif max_index == 2:
            pred_sql.append(bridge_where_data[index]['final_sql'])
        elif max_index == 3:
            pred_sql.append(combination_data[index]['final_sql'])

    output_path = root_path+"lever_verified_sql.txt"
    with open(output_path, 'w') as f:
        for i in pred_sql:
            f.write(i+"\n")
    f.close()
    # input_str = "-- question: List the name, born state and age of the heads of departments ordered by age.| -- SQL:|select name, born state, age from head join management on head.head id = management.head id order by age| |-- exec result:|/*| name born state age| Dudley Hart California 52.0| Jeff Maggert Delaware 53.0|Franklin Langham Connecticut 67.0| Billy Mayfair California 69.0| K. J. Choi Alabama 69.0|*/"

# 仅用训好的verifier来选择正确sql


def xyr_verify_acc(root_path):
    # extra_path = root_path+"0_all_data_with_used_schema.json"
    extra_path = "hard_dev.json"
    bridge_from_path = root_path+"5_bridge_from.json"
    bridge_where_path = root_path+"5_bridge_where.json"
    bridge_not_path = root_path+"5_bridge_not.json"
    combination_path = root_path+"5_combination.json"
    bridge_from_exec_path = root_path+"bridge_from_exec.txt"
    bridge_where_exec_path = root_path+"bridge_where_exec.txt"
    bridge_not_exec_path = root_path+"bridge_not_exec.txt"
    combination_exec_path = root_path+"combination_exec.txt"
    scores_path = root_path+"verify_score.txt"

    with open(extra_path, 'r') as f:
        ori_data = json.load(f)
    f.close()
    ####
    with open(bridge_from_path, 'r') as f:
        bridge_from_data = json.load(f)
    f.close()
    ####
    with open(bridge_where_path, 'r') as f:
        bridge_where_data = json.load(f)
    f.close()
    ####
    with open(bridge_not_path, 'r') as f:
        bridge_not_data = json.load(f)
    f.close()
    ####
    with open(combination_path, 'r') as f:
        combination_data = json.load(f)
    f.close()
    ########################################
    with open(bridge_from_exec_path, 'r') as f:
        bridge_from_exec = f.readlines()
    f.close()
    ####
    with open(bridge_where_exec_path, 'r') as f:
        bridge_where_exec = f.readlines()
    f.close()
    ####
    with open(bridge_not_exec_path, 'r') as f:
        bridge_not_exec = f.readlines()
    f.close()
    ####
    with open(combination_exec_path, 'r') as f:
        combination_exec = f.readlines()
    f.close()

    tokenizer_name = "./model/bert-base-cased"
    model_name = "./model/bert-checkpoint-749"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # prefix = "Check whether the SQL is correct:"
    prefix = ""
    pred_sql = []
    scores = []
    for index, data in enumerate(tqdm(ori_data)):
        # 存放四个的分数
        score = []
        # bridge_from
        input_str = "-- question: "
        input_str += data['question'] + '| '
        input_str += "-- SQL:| "
        input_str += bridge_from_data[index]['final_sql']+"| | "
        input_str += "-- exec result:|/*|"
        # 规则成lever要求格式
        exec_tmp = bridge_from_exec[index].replace("'", "")
        if len(exec_tmp) > 100:
            exec_tmp = exec_tmp[0:100]
        exec_tmp = exec_tmp.replace("), (", " | ")
        exec_tmp = exec_tmp.replace(",", " ")
        if exec_tmp[0] == '[':
            exec_tmp = exec_tmp.replace("[", "")
            exec_tmp = exec_tmp.replace("]", "")
            exec_tmp = exec_tmp.replace("(", "")
            exec_tmp = exec_tmp.replace(")", "")
            exec_tmp = exec_tmp.replace("\n", "")
        input_str += exec_tmp + "|*/"
        # print(input_str)
        encoded_input = tokenizer(prefix+input_str, return_tensors='pt')
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(**encoded_input)
            prediction_logits = outputs.logits
        predictions = torch.softmax(prediction_logits, dim=1).squeeze()
        score.append(float(predictions[1]))
        #################################################################################
        # bridge_not
        input_str = "-- question: "
        input_str += data['question'] + '| '
        input_str += "-- SQL:| "
        input_str += bridge_not_data[index]['final_sql']+"| | "
        input_str += "-- exec result:|/*|"
        # 规则成lever要求格式
        exec_tmp = bridge_not_exec[index].replace("'", "")
        if len(exec_tmp) > 100:
            exec_tmp = exec_tmp[0:100]
        exec_tmp = exec_tmp.replace("), (", " | ")
        exec_tmp = exec_tmp.replace(",", " ")
        if exec_tmp[0] == '[':
            exec_tmp = exec_tmp.replace("[", "")
            exec_tmp = exec_tmp.replace("]", "")
            exec_tmp = exec_tmp.replace("(", "")
            exec_tmp = exec_tmp.replace(")", "")
            exec_tmp = exec_tmp.replace("\n", "")
        input_str += exec_tmp + "|*/"

        encoded_input = tokenizer(prefix+input_str, return_tensors='pt')
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(**encoded_input)
            prediction_logits = outputs.logits
        predictions = torch.softmax(prediction_logits, dim=1).squeeze()
        score.append(float(predictions[1]))
        #################################################################################
        # bridge_where
        input_str = "-- question: "
        input_str += data['question'] + '| '
        input_str += "-- SQL:| "
        input_str += bridge_where_data[index]['final_sql']+"| | "
        input_str += "-- exec result:|/*|"
        # 规则成lever要求格式
        exec_tmp = bridge_where_exec[index].replace("'", "")
        if len(exec_tmp) > 100:
            exec_tmp = exec_tmp[0:100]
        exec_tmp = exec_tmp.replace("), (", " | ")
        exec_tmp = exec_tmp.replace(",", " ")
        if exec_tmp[0] == '[':
            exec_tmp = exec_tmp.replace("[", "")
            exec_tmp = exec_tmp.replace("]", "")
            exec_tmp = exec_tmp.replace("(", "")
            exec_tmp = exec_tmp.replace(")", "")
            exec_tmp = exec_tmp.replace("\n", "")
        input_str += exec_tmp + "|*/"
        encoded_input = tokenizer(prefix+input_str, return_tensors='pt')
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(**encoded_input)
            prediction_logits = outputs.logits
        predictions = torch.softmax(prediction_logits, dim=1).squeeze()
        score.append(float(predictions[1]))
        #################################################################################
        # combination
        input_str = "-- question: "
        input_str += data['question'] + '| '
        input_str += "-- SQL:| "
        input_str += combination_data[index]['final_sql']+"| | "
        input_str += "-- exec result:|/*|"
        # 规则成lever要求格式
        exec_tmp = combination_exec[index].replace("'", "")
        if len(exec_tmp) > 100:
            exec_tmp = exec_tmp[0:100]
        exec_tmp = exec_tmp.replace("), (", " | ")
        exec_tmp = exec_tmp.replace(",", " ")
        if exec_tmp[0] == '[':
            exec_tmp = exec_tmp.replace("[", "")
            exec_tmp = exec_tmp.replace("]", "")
            exec_tmp = exec_tmp.replace("(", "")
            exec_tmp = exec_tmp.replace(")", "")
            exec_tmp = exec_tmp.replace("\n", "")
        input_str += exec_tmp + "|*/"
        encoded_input = tokenizer(prefix+input_str, return_tensors='pt')
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(**encoded_input)
            prediction_logits = outputs.logits
        predictions = torch.softmax(prediction_logits, dim=1).squeeze()
        score.append(float(predictions[1]))
        #################################################################################
        # 选最大值
        max_index = np.argmax(score)
        if max_index == 0:
            pred_sql.append(
                bridge_from_data[index]['final_sql'].replace('\n', ''))
        elif max_index == 1:
            pred_sql.append(
                bridge_not_data[index]['final_sql'].replace('\n', ''))
        elif max_index == 2:
            pred_sql.append(
                bridge_where_data[index]['final_sql'].replace('\n', ''))
        elif max_index == 3:
            pred_sql.append(
                combination_data[index]['final_sql'].replace('\n', ''))
        scores.append(score)

    output_path = root_path+"xyr_bert_verified_sql_10.13.txt"
    with open(output_path, 'w') as f:
        for i in pred_sql:
            f.write(i+"\n")
    f.close()

    # 记录四个的分数
    with open(scores_path, 'w') as f:
        for i in scores:
            f.write(str(i)+"\n")
    # f.close()

    # scores.append(score)

    # input_str = "-- question: List the name, born state and age of the heads of departments ordered by age.| -- SQL:|select name, born state, age from head join management on head.head id = management.head id order by age| |-- exec result:|/*| name born state age| Dudley Hart California 52.0| Jeff Maggert Delaware 53.0|Franklin Langham Connecticut 67.0| Billy Mayfair California 69.0| K. J. Choi Alabama 69.0|*/"

# 加入sql2nl结合verifier来选正确sql


def xyr_choose_correct_new(root_path):
    model = "gpt-4"
    extra_path = "extra_dev.json"
    bridge_from_path = root_path+"5_bridge_from.json"
    bridge_where_path = root_path+"5_bridge_where.json"
    bridge_not_path = root_path+"5_bridge_not.json"
    combination_path = root_path+"5_combination.json"
    bridge_from_exec_path = root_path+"bridge_from_exec.txt"
    bridge_where_exec_path = root_path+"bridge_where_exec.txt"
    bridge_not_exec_path = root_path+"bridge_not_exec.txt"
    combination_exec_path = root_path+"combination_exec.txt"
    scores_path = root_path+"verify_score.txt"

    with open(extra_path, 'r') as f:
        ori_data = json.load(f)
    f.close()
    ####
    with open(bridge_from_path, 'r') as f:
        bridge_from_data = json.load(f)
    f.close()
    ####
    with open(bridge_where_path, 'r') as f:
        bridge_where_data = json.load(f)
    f.close()
    ####
    with open(bridge_not_path, 'r') as f:
        bridge_not_data = json.load(f)
    f.close()
    ####
    with open(combination_path, 'r') as f:
        combination_data = json.load(f)
    f.close()
    ########################################
    with open(bridge_from_exec_path, 'r') as f:
        bridge_from_exec = f.readlines()
    f.close()
    ####
    with open(bridge_where_exec_path, 'r') as f:
        bridge_where_exec = f.readlines()
    f.close()
    ####
    with open(bridge_not_exec_path, 'r') as f:
        bridge_not_exec = f.readlines()
    f.close()
    ####
    with open(combination_exec_path, 'r') as f:
        combination_exec = f.readlines()
    f.close()
    ##################
    with open(scores_path, 'r') as f:
        scores = f.readlines()
    f.close()

    pred_sql = []
    for index, data in enumerate(tqdm(ori_data)):
        # 存放四个的分数
        score = eval(scores[index])
        # print(score)
        #################################################################################
        tmp_exec = []
        tmp_exec.append(bridge_from_exec[index])
        tmp_exec.append(bridge_not_exec[index])
        tmp_exec.append(bridge_where_exec[index])
        tmp_exec.append(combination_exec[index])

        index_map = defaultdict(list)
        for i, string in enumerate(tmp_exec):
            # 如果执行结果error，直接扣分
            if string[0] != '[':
                score[i] -= 100
            # 去掉这些错误的再统计投票
            else:
                index_map[string].append(i)
        duplicates = {string: {"count": len(indices), "indices": indices}
                      for string, indices in index_map.items() if len(indices) > 1}
        # 如果都小于0.5,即全部低置信度
        if max(score) < 0.5:
            # 对于四个候选，只有一个最大值才使用投票，否则2:2平
            # print(duplicates)
            if duplicates and len(duplicates) == 1:
                for string, info in duplicates.items():
                    max_index = info['indices'][0]
            # 没法通过投票来解决，因此采用转nl再判断语意的方式
            # else:
            #     candidate_sql = []
            #     candidate_sql.append(bridge_from_data[index]['final_sql'])
            #     candidate_sql.append(bridge_not_data[index]['final_sql'])
            #     candidate_sql.append(bridge_where_data[index]['final_sql'])
            #     candidate_sql.append(combination_data[index]['final_sql'])

            #     for can_index, candidate in enumerate(candidate_sql):
            #         prompt = "Please carefully write the following natural language question corresponding to the SQL, only the corresponding question, do not analyze\n"
            #         prompt += "SQL:"
            #         prompt += candidate
            #         prompt += "\nQuestion:"
            #         n_repeat = 0
            #         # print(prompt)
            #         while True:
            #             try:
            #                 res = chat_gpt(model, prompt)
            #                 break
            #             except Exception as e:
            #                 n_repeat += 1
            #                 print(
            #                     f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
            #                 time.sleep(1)
            #                 continue
            #         print('result:'+str(res))
            #         nl = res['response'][0]

            #         prompt = "Determine if the two natural language questions below mean the same thing, and answer yes or no without giving any analysis\n"
            #         prompt += "Question1:"
            #         prompt += nl
            #         prompt += "\nQuestion2:"
            #         prompt += bridge_from_data[index]['ori_question']
            #         n_repeat = 0
            #         while True:
            #             try:
            #                 res = chat_gpt(model, prompt)
            #                 break
            #             except Exception as e:
            #                 n_repeat += 1
            #                 print(
            #                     f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
            #                 time.sleep(1)
            #                 continue
            #         print('result:'+str(res))
            #         consistency = res['response'][0]

            #         if consistency[:3] == "yes" or consistency == "Yes":
            #             score[can_index] += 0.5

            #     max_index = np.argmax(score)
            else:
                max_index = np.argmax(score)
        # 否则，需要看有几个高置信度，如果多个高置信度答案是否都一样
        else:
            high_conf = []
            # 找出高置信度对应的编号
            for s_index, s in enumerate(score):
                if s > 0.5:
                    high_conf.append(s_index)
            # 有多个高置信度的，检查是否一致
            if len(high_conf) > 1:
                tmp = tmp_exec[high_conf[0]]
                flag = 1
                for h in high_conf:
                    if tmp != tmp_exec[h]:
                        flag = 0
                        break
                if flag == 0:
                    candidate_sql = []
                    for h in high_conf:
                        if h == 0:
                            candidate_sql.append(
                                bridge_from_data[index]['final_sql'])
                        elif h == 1:
                            candidate_sql.append(
                                bridge_not_data[index]['final_sql'])
                        elif h == 2:
                            candidate_sql.append(
                                bridge_where_data[index]['final_sql'])
                        elif h == 3:
                            candidate_sql.append(
                                combination_data[index]['final_sql'])

                    for can_index, candidate in enumerate(candidate_sql):
                        prompt = "Please carefully write the following natural language question corresponding to the SQL, only the corresponding question, do not analyze\n"
                        prompt += "SQL:"
                        prompt += candidate
                        prompt += "\nQuestion:"
                        n_repeat = 0
                        # print(prompt)
                        while True:
                            try:
                                res = chat_gpt(model, prompt)
                                break
                            except Exception as e:
                                n_repeat += 1
                                print(
                                    f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                                time.sleep(1)
                                continue
                        print('result:'+str(res))
                        nl = res['response'][0]

                        prompt = "Determine if the two natural language questions below mean the same thing, and answer yes or no without giving any analysis\n"
                        prompt += "Question1:"
                        prompt += nl
                        prompt += "\nQuestion2:"
                        prompt += bridge_from_data[index]['ori_question']
                        n_repeat = 0
                        while True:
                            try:
                                res = chat_gpt(model, prompt)
                                break
                            except Exception as e:
                                n_repeat += 1
                                print(
                                    f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                                time.sleep(1)
                                continue
                        print('result:'+str(res))
                        consistency = res['response'][0]

                        if consistency[:3] == "yes" or consistency == "Yes":
                            score[high_conf[can_index]] += 0.5
            max_index = np.argmax(score)

        assert max_index in [0, 1, 2, 3]
        if max_index == 0:
            pred_sql.append(bridge_from_data[index]['final_sql'])
        elif max_index == 1:
            pred_sql.append(bridge_not_data[index]['final_sql'])
        elif max_index == 2:
            pred_sql.append(bridge_where_data[index]['final_sql'])
        elif max_index == 3:
            pred_sql.append(combination_data[index]['final_sql'])

    output_path = root_path+"xyr_new_verified_sql.txt"
    with open(output_path, 'w') as f:
        for i in pred_sql:
            f.write(i+"\n")
    f.close()

# 加入sql2nl以及consistency内容的verifier


def xyr_sql2nl_verify(root_path):
    model = "gpt-4"
    extra_path = root_path+"0_all_data_with_used_schema.json"
    bridge_from_path = root_path+"5_bridge_from.json"
    bridge_where_path = root_path+"5_bridge_where.json"
    bridge_not_path = root_path+"5_bridge_not.json"
    combination_path = root_path+"5_combination.json"
    bridge_from_exec_path = root_path+"bridge_from_exec.txt"
    bridge_where_exec_path = root_path+"bridge_where_exec.txt"
    bridge_not_exec_path = root_path+"bridge_not_exec.txt"
    combination_exec_path = root_path+"combination_exec.txt"
    scores_path = root_path+"verify_score.txt"

    with open(extra_path, 'r') as f:
        ori_data = json.load(f)
    f.close()
    ####
    with open(bridge_from_path, 'r') as f:
        bridge_from_data = json.load(f)
    f.close()
    ####
    with open(bridge_where_path, 'r') as f:
        bridge_where_data = json.load(f)
    f.close()
    ####
    with open(bridge_not_path, 'r') as f:
        bridge_not_data = json.load(f)
    f.close()
    ####
    with open(combination_path, 'r') as f:
        combination_data = json.load(f)
    f.close()
    ########################################
    with open(bridge_from_exec_path, 'r') as f:
        bridge_from_exec = f.readlines()
    f.close()
    ####
    with open(bridge_where_exec_path, 'r') as f:
        bridge_where_exec = f.readlines()
    f.close()
    ####
    with open(bridge_not_exec_path, 'r') as f:
        bridge_not_exec = f.readlines()
    f.close()
    ####
    with open(combination_exec_path, 'r') as f:
        combination_exec = f.readlines()
    f.close()
    ##################
    with open(scores_path, 'r') as f:
        scores = f.readlines()
    f.close()

    pred_sql = []
    feedback = []
    for index, data in enumerate(tqdm(ori_data)):
        tmp_feedback = []
        # 存放四个的分数
        score = eval(scores[index])
        #################################################################################
        tmp_exec = []
        tmp_exec.append(bridge_from_exec[index])
        tmp_exec.append(bridge_not_exec[index])
        tmp_exec.append(bridge_where_exec[index])
        tmp_exec.append(combination_exec[index])

        index_map = defaultdict(list)
        for i, string in enumerate(tmp_exec):
            # # 如果执行结果error，直接扣分
            # if string[0] != '[':
            #     score[i] -= 100
            # # 去掉这些错误的再统计投票
            # else:
            #     index_map[string].append(i)
            index_map[string].append(i)
        # 记录的是含有多个相同执行结果的candidate
        duplicates = {string: {"count": len(indices), "indices": indices}
                      for string, indices in index_map.items() if len(indices) > 1}
        # 如果都小于0.9,即全部低置信度
        if max(score) < 0.9:
            # 对于四个候选，只有一个最大值才使用投票，否则2:2平
            # print(duplicates)
            # if duplicates and len(duplicates) == 1:
            #     for string, info in duplicates.items():
            #         max_index = info['indices'][0]
            # 没法通过投票来解决，因此采用转nl再判断语意的方式
            # else:
            candidate_sql = []
            candidate_sql.append(bridge_from_data[index]['final_sql'])
            candidate_sql.append(bridge_not_data[index]['final_sql'])
            candidate_sql.append(bridge_where_data[index]['final_sql'])
            candidate_sql.append(combination_data[index]['final_sql'])
            assert len(candidate_sql) == 4
            for can_index, candidate in enumerate(candidate_sql):
                prompt = "Please carefully write the following natural language question corresponding to the SQL without neglecting semantic details. Only return the corresponding question, do not analyze.\n"
                prompt += "SQL:"
                prompt += candidate
                prompt += "\nQuestion:"
                n_repeat = 0
                # print(prompt)
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                nl = res['response'][0]

                prompt = "Determine if the two natural language questions below mean the same thing, and answer yes or no and give analysis.\nHere is an example:\n"
                prompt += "Question1: What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?\n"
                prompt += "Question2: Which stadiums held concerts in 2015 but did not hold concerts in 2014?\n"
                prompt += "consistency: no\n"
                prompt += "analysis: Question1 is looking for stadiums that had concerts in both years, whereas Question2 is looking for stadiums that had concerts only in 2015 and not in 2014.\n\n"
                prompt += "Let's begin:\n"
                prompt += "Question1:"
                prompt += bridge_from_data[index]['question']
                prompt += "\nQuestion2:"
                prompt += nl
                n_repeat = 0
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                response = res['response'][0].lower()
                response = response.replace('\n', '')
                try:
                    analysis = response.split('analysis: ')[1].strip()
                except:
                    analysis = response
                try:
                    consistency = response.split('analysis:')[0]
                    if 'yes' in consistency:
                        consistency = 'yes'
                    else:
                        consistency = 'no'
                except:
                    if 'yes' in response:
                        consistency = 'yes'
                    else:
                        consistency = 'no'
                # 给一致的加50分，对于那些执行无error的，就可以得到很高的分数
                # print('<')
                # print(consistency)
                # print(analysis)
                # print('>')
                assert consistency in ['yes', 'no']
                if 'yes' in consistency or 'Yes' in consistency:
                    score[can_index] += 50
                tmp_feedback.append(consistency+'\t'+analysis)
            # 选出了最高分的候选sql
            max_index = np.argmax(score)
            feedback.append(tmp_feedback[max_index])
            # else:
            #     max_index = np.argmax(score)
        # 否则，需要看有几个高置信度，如果多个高置信度答案是否都一样
        else:
            high_conf = []
            # 找出高置信度对应的编号
            for s_index, s in enumerate(score):
                if s > 0.9:
                    high_conf.append(s_index)
            # 有多个高置信度的，检查是否一致；否则意味着只有一个高分的，那么就直接选出最高分的输出
            if len(high_conf) > 1:
                tmp = tmp_exec[high_conf[0]]
                flag = 1
                for h in high_conf:
                    if tmp != tmp_exec[h]:
                        flag = 0
                        break
                # 说明存在不一致
                if flag == 0:
                    candidate_sql = []
                    for h in high_conf:
                        if h == 0:
                            candidate_sql.append(
                                bridge_from_data[index]['final_sql'])
                        elif h == 1:
                            candidate_sql.append(
                                bridge_not_data[index]['final_sql'])
                        elif h == 2:
                            candidate_sql.append(
                                bridge_where_data[index]['final_sql'])
                        elif h == 3:
                            candidate_sql.append(
                                combination_data[index]['final_sql'])
                    assert len(high_conf) == len(candidate_sql)
                    for can_index, candidate in enumerate(candidate_sql):
                        prompt = "Please carefully write the following natural language question corresponding to the SQL, only the corresponding question, do not analyze\n"
                        prompt += "SQL:"
                        prompt += candidate
                        prompt += "\nQuestion:"
                        n_repeat = 0
                        # print(prompt)
                        while True:
                            try:
                                res = chat_gpt(model, prompt)
                                break
                            except Exception as e:
                                n_repeat += 1
                                print(
                                    f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                                time.sleep(1)
                                continue
                        print('result:'+str(res))
                        nl = res['response'][0]

                        prompt = "Determine if the two natural language questions below mean the same thing, and answer yes or no and give analysis.\nHere is an example:\n"
                        prompt += "Question1: What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?\n"
                        prompt += "Question2: Which stadiums held concerts in 2015 but did not hold concerts in 2014?\n"
                        prompt += "consistency: no\n"
                        prompt += "analysis: Question1 is looking for stadiums that had concerts in both years, whereas Question2 is looking for stadiums that had concerts only in 2015 and not in 2014.\n\n"
                        prompt += "Let's begin:\n"
                        prompt += "Question1:"
                        prompt += bridge_from_data[index]['question']
                        prompt += "\nQuestion2:"
                        prompt += nl
                        n_repeat = 0
                        while True:
                            try:
                                res = chat_gpt(model, prompt)
                                break
                            except Exception as e:
                                n_repeat += 1
                                print(
                                    f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                                time.sleep(1)
                                continue
                        print('result:'+str(res))
                        response = res['response'][0].lower()
                        response = response.replace('\n', '')
                        try:
                            analysis = response.split('analysis:')[1].strip()
                        except:
                            analysis = response
                        try:
                            consistency = response.split('analysis:')[0]
                            if 'yes' in consistency:
                                consistency = 'yes'
                            else:
                                consistency = 'no'
                        except:
                            if 'yes' in response:
                                consistency = 'yes'
                            else:
                                consistency = 'no'
                        assert consistency in ['yes', 'no']
                        # 给一致的加50分，对于那些执行无error的，就可以得到很高的分数
                        if 'yes' in consistency or 'Yes' in consistency:
                            score[high_conf[can_index]] += 50
                        tmp_feedback.append(consistency+'\t'+analysis)
                    assert len(tmp_feedback) == len(high_conf)
                    max_index = np.argmax(score)
                    feedback.append(tmp_feedback[high_conf.index(max_index)])
                # 全一致，经过尝试，不做语义一致性分析比较好
                else:
                    max_index = np.argmax(score)
                    # can_index = np.argmax(score)
                    # if can_index == 0:
                    #     candidate = bridge_from_data[index]['final_sql']
                    # elif can_index == 1:
                    #     candidate = bridge_not_data[index]['final_sql']
                    # elif can_index == 2:
                    #     candidate = bridge_where_data[index]['final_sql']
                    # elif can_index == 3:
                    #     candidate = combination_data[index]['final_sql']
                    # prompt = "Please carefully write the following natural language question corresponding to the SQL, only the corresponding question, do not analyze\n"
                    # prompt += "SQL:"
                    # prompt += candidate
                    # prompt += "\nQuestion:"
                    # n_repeat = 0

                    # while True:
                    #     try:
                    #         res = chat_gpt(model, prompt)
                    #         break
                    #     except Exception as e:
                    #         n_repeat += 1
                    #         print(
                    #             f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                    #         time.sleep(1)
                    #         continue
                    # print('result:'+str(res))
                    # nl = res['response'][0]

                    # prompt = "Determine if the two natural language questions below mean the same thing, and answer yes or no and give analysis.\nHere is an example:\n"
                    # prompt += "Question1: What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?\n"
                    # prompt += "Question2: Which stadiums held concerts in 2015 but did not hold concerts in 2014?\n"
                    # prompt += "consistency: no\n"
                    # prompt += "analysis: Question1 is looking for stadiums that had concerts in both years, whereas Question2 is looking for stadiums that had concerts only in 2015 and not in 2014.\n\n"
                    # prompt += "Let's begin:\n"
                    # prompt += "Question1:"
                    # prompt += bridge_from_data[index]['ori_question']
                    # prompt += "\nQuestion2:"
                    # prompt += nl
                    # n_repeat = 0
                    # while True:
                    #     try:
                    #         res = chat_gpt(model, prompt)
                    #         break
                    #     except Exception as e:
                    #         n_repeat += 1
                    #         print(
                    #             f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                    #         time.sleep(1)
                    #         continue
                    # print('result:'+str(res))
                    # response = res['response'][0].lower()
                    # response = response.replace('\n','')
                    # try:
                    #     analysis = response.split('analysis:')[1].strip()
                    # except:
                    #     analysis = response
                    # try:
                    #     consistency = response.split('analysis:')[0]
                    #     if 'yes' in consistency:
                    #         consistency = 'yes'
                    #     else:
                    #         consistency = 'no'
                    # except:
                    #     if 'yes' in response:
                    #         consistency = 'yes'
                    #     else:
                    #         consistency = 'no'
                    # assert consistency in ['yes','no']
                    # 给一致的加50分，对于那些执行无error的，就可以得到很高的分数
                    # if 'yes' in consistency or 'Yes' in consistency:
                    #     score[can_index] += 50
                    # feedback.append(consistency+'\t'+analysis)
                    feedback.append('')
            # 只有一个高置信度，经过尝试，不做语义一致性分析比较好
            else:
                max_index = np.argmax(score)
                # can_index = np.argmax(score)
                # if can_index == 0:
                #     candidate = bridge_from_data[index]['final_sql']
                # elif can_index == 1:
                #     candidate = bridge_not_data[index]['final_sql']
                # elif can_index == 2:
                #     candidate = bridge_where_data[index]['final_sql']
                # elif can_index == 3:
                #     candidate = combination_data[index]['final_sql']
                # prompt = "Please carefully write the following natural language question corresponding to the SQL, only the corresponding question, do not analyze\n"
                # prompt += "SQL:"
                # prompt += candidate
                # prompt += "\nQuestion:"
                # n_repeat = 0
                # while True:
                #     try:
                #         res = chat_gpt(model, prompt)
                #         break
                #     except Exception as e:
                #         n_repeat += 1
                #         print(
                #             f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                #         time.sleep(1)
                #         continue
                # print('result:'+str(res))
                # nl = res['response'][0]

                # prompt = "Determine if the two natural language questions below mean the same thing, and answer yes or no and give analysis.\nHere is an example:\n"
                # prompt += "Question1: What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015?\n"
                # prompt += "Question2: Which stadiums held concerts in 2015 but did not hold concerts in 2014?\n"
                # prompt += "consistency: no\n"
                # prompt += "analysis: Question1 is looking for stadiums that had concerts in both years, whereas Question2 is looking for stadiums that had concerts only in 2015 and not in 2014.\n\n"
                # prompt += "Let's begin:\n"
                # prompt += "Question1:"
                # prompt += bridge_from_data[index]['ori_question']
                # prompt += "\nQuestion2:"
                # prompt += nl
                # n_repeat = 0
                # while True:
                #     try:
                #         res = chat_gpt(model, prompt)
                #         break
                #     except Exception as e:
                #         n_repeat += 1
                #         print(
                #             f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                #         time.sleep(1)
                #         continue
                # print('result:'+str(res))
                # response = res['response'][0].lower()
                # response = response.replace('\n','')
                # try:
                #     analysis = response.split('analysis:')[1].strip()
                # except:
                #     analysis = response
                # try:
                #     consistency = response.split('analysis:')[0]
                #     if 'yes' in consistency:
                #         consistency = 'yes'
                #     else:
                #         consistency = 'no'
                # except:
                #     if 'yes' in response:
                #         consistency = 'yes'
                #     else:
                #         consistency = 'no'
                # assert consistency in ['yes','no']
                # 给一致的加50分，对于那些执行无error的，就可以得到很高的分数
                # if 'yes' in consistency or 'Yes' in consistency:
                #     score[can_index] += 50
                # feedback.append(consistency+'\t'+analysis)
                feedback.append('')

        assert max_index in [0, 1, 2, 3]
        if max_index == 0:
            pred_sql.append(bridge_from_data[index]['final_sql'])
        elif max_index == 1:
            pred_sql.append(bridge_not_data[index]['final_sql'])
        elif max_index == 2:
            pred_sql.append(bridge_where_data[index]['final_sql'])
        elif max_index == 3:
            pred_sql.append(combination_data[index]['final_sql'])

        output_path = root_path+"xyr_sql2nl_verified_sql.txt"
        with open(output_path, 'w') as f:
            for i in pred_sql:
                i = i.replace('\n', '')
                f.write(i+"\n")
        output_path = root_path+"xyr_analysis.txt"
        with open(output_path, 'w') as f:
            for i in feedback:
                f.write(i+"\n")

    # output_path = root_path+"xyr_9.20_sql2nl_verified_sql.txt"
    # with open(output_path, 'w') as f:
    #     for i in pred_sql:
    #         f.write(i+"\n")
    # output_path = root_path+"xyr_9.20_analysis.txt"
    # with open(output_path, 'w') as f:
    #     for i in feedback:
    #         f.write(i+"\n")

# 提供备选的SQL以及一致性与区别，进行修正


def xyr_revision(root_path):
    # test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database",gold=root_path+"6_gold_bridge_from.txt",predict=root_path+"xyr_sql2nl_verified_sql.txt",etype="exec",exec_result_path=root_path+"exec_result.txt",plug_value=False,kmaps=None,keep_distinct=False,progress_bar_for_each_datapoint=False)

    model = "gpt-4o"
    extra_path = INPUT_PATH
    schema_path = root_path+"c3_schema.json"
    exec_result_path = root_path+"exec_result.txt"
    analysis_path = root_path+"xyr_analysis.txt"
    sql_path = root_path+"xyr_sql2nl_verified_sql.txt"

    with open(extra_path, 'r') as f:
        datas = json.load(f)
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    with open(exec_result_path, 'r') as f:
        exec_result = f.readlines()
    with open(analysis_path, 'r') as f:
        analysis_result = f.readlines()
    with open(sql_path, 'r') as f:
        ori_sql = f.readlines()
    verified_sql = []
    for index, data in enumerate(tqdm(datas)):
        assert len(verified_sql) == index
        # 如果执行存在error
        if exec_result[index][0] != '[':
            # 如果语义判断不一致
            if analysis_result[index][:2] == 'no':
                prompt = "## You are an NL2SQL expert. For the given SQL statement, I need you to use the relevant information to correct it. \
                    The information provided includes the SQL statement, its corresponding natural language, execution error messages, \
                    descriptions of the differences between the SQL and the natural language, and the database schema information. \
                    ## You need to first use diffences between SQL and Question to check whether the overall structure of the original SQL is correct, mainly including the use of keywords. \
                    Then check whether the table name, column name, and joins of the table are correct according to execution error.\
                    Please output the modified SQL statement directly, without any explanation.\n"
                prompt += "Original SQL: "
                prompt += ori_sql[index] + '\n'
                prompt += "Question: "
                prompt += data['question']+'\n'
                prompt += "execution error: "
                prompt += exec_result[index]+'\n'
                prompt += "diffences between SQL and Question: "
                analysis = analysis_result[index].split('\t')[1]
                analysis = analysis.replace('question1', 'Question')
                analysis = analysis.replace('question2', 'SQL')
                prompt += analysis+'\n'
                prompt += "schema: "
                prompt += schema[index]['input_sequence']+'\n'
                prompt = prompt.replace('    ', '')
                prompt = prompt.replace('\t', '')
                n_repeat = 0
                print(prompt)
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                response = res['response'][0]
                response = response.replace('\n', ' ')
                response += '\n'
                verified_sql.append(response)
            # 如果语义判断一致
            else:
                prompt = "## You are an NL2SQL expert. For the given SQL statement, I need you to use the relevant information to correct it. \
                    The information provided includes the SQL statement, its corresponding natural language, execution error messages, \
                    and the database schema information. \
                    ## The original SQL structure, such as keywords, is basically correct. You may need to check the correctness of the table name, column name and joins of the table. \
                    Please analyze specific errors with execution error messages. \
                    ## Please output the modified SQL statement directly, without any explanation.\n"
                prompt += "Original SQL: "
                prompt += ori_sql[index] + '\n'
                prompt += "Question: "
                prompt += data['question']+'\n'
                prompt += "execution error: "
                prompt += exec_result[index]+'\n'
                prompt += "schema: "
                prompt += schema[index]['input_sequence']+'\n'
                prompt = prompt.replace('    ', '')
                prompt = prompt.replace('\t', '')
                # print(prompt)
                n_repeat = 0
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                response = res['response'][0]
                response = response.replace('\n', ' ')
                response += '\n'
                verified_sql.append(response)
        # 不存在error
        else:
            # 如果语义判断不一致
            if analysis_result[index][:2] == 'no':
                prompt = "## You are an NL2SQL expert. For the given SQL statement, I need you to use the relevant information to correct it. \
                    The information provided includes the SQL statement, its corresponding natural language, \
                    descriptions of the differences between the SQL and the natural language, and the database schema information. \n\
                    ## You need to use diffences between SQL and Question to check whether the overall structure of the original SQL is correct, mainly including the use of keywords. \
                    Please output the modified SQL statement directly, without any explanation.\n"
                prompt += "Original SQL: "
                prompt += ori_sql[index] + '\n'
                prompt += "Question: "
                prompt += data['question']+'\n'
                prompt += "diffences between SQL and Question: "
                analysis = analysis_result[index].split('\t')[1]
                analysis = analysis.replace('question1', 'Question')
                analysis = analysis.replace('question2', 'SQL')
                prompt += analysis+'\n'
                prompt += "schema: "
                prompt += schema[index]['input_sequence']+'\n'
                prompt = prompt.replace('    ', '')
                prompt = prompt.replace('\t', '')
                n_repeat = 0
                print(prompt)
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                response = res['response'][0]
                response = response.replace('\n', ' ')
                response += '\n'
                verified_sql.append(response)
            # 如果语义判断一致, 无附加信息直接输出
            else:
                prompt = "## You are an NL2SQL expert. For the given SQL and corresponding Question, I need you to check whether there are any errors in the SQL. \
                If yes, correct the original SQL and output the corrected SQL. Else, output the original SQL directly without any explanation.\n"
                prompt += "Original SQL: "
                prompt += ori_sql[index]
                prompt += "Question: "
                prompt += data['question']+'\n'
                prompt += "schema:\n"
                prompt += schema[index]['input_sequence']+'\n'
                prompt += "corrected SQL: "
                prompt = prompt.replace('    ', '')
                prompt = prompt.replace('\t', '')
                n_repeat = 0
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                response = res['response'][0]
                response = response.replace('\n', ' ')
                response += '\n'
                verified_sql.append(response)

    output_path = root_path+"revision_sql.txt"
    with open(output_path, 'w') as f:
        for i in verified_sql:
            f.write(i)

    # test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database",gold=root_path+"6_gold_bridge_from.txt",predict=root_path+"9.20/xyr_9.20_sql2nl_verified_sql.txt",etype="exec",exec_result_path=root_path+"9.20/exec_result.txt",plug_value=False,kmaps=None,keep_distinct=False,progress_bar_for_each_datapoint=False)

# 使用真实的schema进行revision


def xyr_revision_with_schema(root_path):
    test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database", gold=root_path+"6_gold_bridge_from.txt", predict=root_path+"xyr_sql2nl_verified_sql.txt",
                          etype="exec", exec_result_path=root_path+"exec_result.txt", plug_value=False, kmaps=None, keep_distinct=False, progress_bar_for_each_datapoint=False)

    model = "gpt-4"
    extra_path = root_path+"0_all_data_with_used_schema.json"
    # extra_path_1 = root_path+"0_data_need_decompose.json"
    # extra_path_2 = root_path+"0_data_no_need_decompose.json"
    # schema_path = root_path+"9.23/c3_schema.json"
    exec_result_path = root_path+"exec_result.txt"
    analysis_path = root_path+"xyr_analysis.txt"
    sql_path = root_path+"xyr_sql2nl_verified_sql.txt"

    with open(extra_path, 'r') as f:
        datas = json.load(f)
    # with open(extra_path_1, 'r') as f:
    #     datas = json.load(f)
    # with open(extra_path_2, 'r') as f:
    #     datas += json.load(f)
    # with open(schema_path, 'r') as f:
    #     schema = json.load(f)
    with open(exec_result_path, 'r') as f:
        exec_result = f.readlines()
    with open(analysis_path, 'r') as f:
        analysis_result = f.readlines()
    with open(sql_path, 'r') as f:
        ori_sql = f.readlines()
    verified_sql = []
    for index, data in enumerate(tqdm(datas)):
        assert len(verified_sql) == index
        # 如果执行存在error
        if exec_result[index][0] != '[':
            # 如果语义判断不一致
            if analysis_result[index][:2] == 'no':
                prompt = "## You are an NL2SQL expert. For the given SQL statement, I need you to use the relevant information to correct it. \
                    The information provided includes the SQL statement, its corresponding natural language, execution error messages, \
                    descriptions of the differences between the SQL and the natural language, and the database schema information. \
                    ## You need to first use diffences between SQL and Question to check whether the overall structure of the original SQL is correct, mainly including the use of keywords. \
                    Then check whether the table name, column name, and joins of the table are correct according to execution error.\
                    Please output the modified SQL statement directly, without any explanation.\n"
                prompt += "Original SQL: "
                prompt += ori_sql[index] + '\n'
                prompt += "Question: "
                prompt += data['question']+'\n'
                prompt += "execution error: "
                prompt += exec_result[index]+'\n'
                prompt += "diffences between SQL and Question: "
                analysis = analysis_result[index].split('\t')[1]
                analysis = analysis.replace('question1', 'Question')
                analysis = analysis.replace('question2', 'SQL')
                prompt += analysis+'\n'
                prompt += "schema: "
                schema_str = ""
                for table in data['used_schema']['tables'].keys():
                    schema_str += table + '('
                    for column in data['used_schema']['tables'][table]:
                        schema_str += column + ', '
                    schema_str = schema_str[:-2] + ')\n'
                for fk in data['used_schema']['fks']:
                    schema_str += fk + '\n'
                prompt += schema_str
                prompt = prompt.replace('    ', '')
                prompt = prompt.replace('\t', '')
                n_repeat = 0
                print(prompt)
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                response = res['response'][0]
                response = response.replace('\n', ' ')
                response += '\n'
                verified_sql.append(response)
            # 如果语义判断一致
            else:
                prompt = "## You are an NL2SQL expert. For the given SQL statement, I need you to use the relevant information to correct it. \
                    The information provided includes the SQL statement, its corresponding natural language, execution error messages, \
                    and the database schema information. \
                    ## The original SQL structure, such as keywords, is basically correct. You may need to check the correctness of the table name, column name and joins of the table. \
                    Please analyze specific errors with execution error messages. \
                    ## Please output the modified SQL statement directly, without any explanation.\n"
                prompt += "Original SQL: "
                prompt += ori_sql[index] + '\n'
                prompt += "Question: "
                prompt += data['question']+'\n'
                prompt += "execution error: "
                prompt += exec_result[index]+'\n'
                prompt += "schema: "
                schema_str = ""
                for table in data['used_schema']['tables'].keys():
                    schema_str += table + '('
                    for column in data['used_schema']['tables'][table]:
                        schema_str += column + ', '
                    schema_str = schema_str[:-2] + ')\n'
                for fk in data['used_schema']['fks']:
                    schema_str += fk + '\n'
                prompt += schema_str
                prompt = prompt.replace('    ', '')
                prompt = prompt.replace('\t', '')
                # print(prompt)
                n_repeat = 0
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                response = res['response'][0]
                response = response.replace('\n', ' ')
                response += '\n'
                verified_sql.append(response)
        # 不存在error
        else:
            # 如果语义判断不一致
            if analysis_result[index][:2] == 'no':
                prompt = "## You are an NL2SQL expert. For the given SQL statement, I need you to use the relevant information to correct it. \
                    The information provided includes the SQL statement, its corresponding natural language, \
                    descriptions of the differences between the SQL and the natural language, and the database schema information. \
                    ## You need to use diffences between SQL and Question to check whether the overall structure of the original SQL is correct, mainly including the use of keywords. \
                    Please output the modified SQL statement directly, without any explanation.\n"
                prompt += "Original SQL: "
                prompt += ori_sql[index] + '\n'
                prompt += "Question: "
                prompt += data['question']+'\n'
                prompt += "diffences between SQL and Question: "
                analysis = analysis_result[index].split('\t')[1]
                analysis = analysis.replace('question1', 'Question')
                analysis = analysis.replace('question2', 'SQL')
                prompt += analysis+'\n'
                prompt += "schema: "
                schema_str = ""
                for table in data['used_schema']['tables'].keys():
                    schema_str += table + '('
                    for column in data['used_schema']['tables'][table]:
                        schema_str += column + ', '
                    schema_str = schema_str[:-2] + ')\n'
                for fk in data['used_schema']['fks']:
                    schema_str += fk + '\n'
                prompt += schema_str
                prompt = prompt.replace('    ', '')
                prompt = prompt.replace('\t', '')
                n_repeat = 0
                print(prompt)
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                response = res['response'][0]
                response = response.replace('\n', ' ')
                response += '\n'
                verified_sql.append(response)
            # 如果语义判断一致, 无附加信息直接输出
            else:
                prompt = "## You are an NL2SQL expert. For the given SQL and corresponding Question, I need you to check whether there are any errors in the SQL. \
                If yes, correct the original SQL and output the corrected SQL. Else, output the original SQL directly without any explanation.\n"
                prompt += "Original SQL: "
                prompt += ori_sql[index]
                prompt += "Question: "
                prompt += data['question']+'\n'
                prompt += "schema:\n"
                schema_str = ""
                for table in data['used_schema']['tables'].keys():
                    schema_str += table + '('
                    for column in data['used_schema']['tables'][table]:
                        schema_str += column + ', '
                    schema_str = schema_str[:-2] + ')\n'
                for fk in data['used_schema']['fks']:
                    schema_str += fk + '\n'
                prompt += schema_str
                prompt += "corrected SQL: "
                prompt = prompt.replace('    ', '')
                prompt = prompt.replace('\t', '')
                n_repeat = 0
                while True:
                    try:
                        res = chat_gpt(model, prompt)
                        break
                    except Exception as e:
                        n_repeat += 1
                        print(
                            f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                        time.sleep(1)
                        continue
                print('result:'+str(res))
                response = res['response'][0]
                response = response.replace('\n', ' ')
                response += '\n'
                verified_sql.append(response)

    output_path = root_path+"revision_sql.txt"
    with open(output_path, 'w') as f:
        for i in verified_sql:
            f.write(i)

    # test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database",gold=root_path+"6_gold_bridge_from.txt",predict=root_path+"9.20/xyr_9.20_sql2nl_verified_sql.txt",etype="exec",exec_result_path=root_path+"9.20/exec_result.txt",plug_value=False,kmaps=None,keep_distinct=False,progress_bar_for_each_datapoint=False)

# 以下为10.3重新写的verify revision一体结构
# 返回bert打分


def bert_verifier_for_score(tokenizer, model, question, sql, exec_result):
    # bridge_not
    input_str = "-- question: "
    input_str += question + '| '
    input_str += "-- SQL:| "
    input_str += sql+"| | "
    input_str += "-- exec result:|/*|"
    # 规则成lever要求格式
    exec_tmp = exec_result.replace("'", "")
    if len(exec_tmp) > 100:
        exec_tmp = exec_tmp[0:100]
    exec_tmp = exec_tmp.replace("), (", " | ")
    exec_tmp = exec_tmp.replace(",", " ")
    if exec_tmp[0] == '[':
        exec_tmp = exec_tmp.replace("[", "")
        exec_tmp = exec_tmp.replace("]", "")
        exec_tmp = exec_tmp.replace("(", "")
        exec_tmp = exec_tmp.replace(")", "")
        exec_tmp = exec_tmp.replace("\n", "")
    input_str += exec_tmp + "|*/"
    # print(input_str)
    encoded_input = tokenizer(
        input_str, return_tensors='pt', max_length=512, truncation=True)
    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(**encoded_input)
        prediction_logits = outputs.logits
    predictions = torch.softmax(prediction_logits, dim=1).squeeze()
    return float(predictions[1])
    # score.append(float(predictions[1]))


def sql_execute(db_name, sql):
    db_path = '/home3/xianyiran/text2sql/spider/database/'
    db_name = db_name
    path = db_path+f"{db_name}/{db_name}.sqlite"
    # print(path)
    conn = sqlite3.connect(path)
    cs = conn.cursor()
    try:
        cs.execute(sql)
        result = cs.fetchall()
    except sqlite3.Error as e:
        result = e
    return str(result)

# 使用分类的revision


def xyr_classified_revision(root_path):
    # llm_model = "gpt-4o-2024-08-06"
    llm_model = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model('gpt-4')
    total_tokens = 0
    total_response_tokens = 0
    data_path = root_path+"all_candidate_sql.json"
    with open(data_path, 'r') as f:
        datas = json.load(f)
    all_schema_path = root_path+"0_all_schema.json"
    with open(all_schema_path, 'r') as f:
        all_schemas = json.load(f)

    tokenizer_name = "./model/bert-base-cased"
    model_name = "./model/bert-checkpoint-749"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    PROMPT_SELECT_CASE = "## You are a text2sql master. Given an SQL statement, the corresponding natural language question, and the associated database schema information, \
    I need you to determine whether the meaning expressed by the SQL statement is exactly consistent with the natural language problem. \
    I will give you several cases of semantic inconsistencies, if there is at least one of them in SQL, then the semantics are inconsistent, \
    otherwise the semantics are consistent. When the semantic of SQL and natural language questions are not consistent, \
    output the corresponding number of the case contained in the SQL. \n\
    ## Here are the error cases:\n\
    # case1: The ORDER BY keyword is conditional on multiple columns.\n\
    # case2: The SQL query has WHERE clause or JOIN conditions where the columns don't follow a correct primary key-foreign key relationship.\n\
    # case3: Some tables that are clearly related to the question do not appear when they should appear in SQL.\n\n\
    ## Here are some examples:\n\n\
    Question: What is the average age of students who do not have any pet.\n\
    SQL: SELECT avg(age) FROM Student WHERE StuID NOT IN (SELECT StuID FROM Has_Pet)\n\
    database schema:\n\
    # student ( stuid, age, lname, fname )\n# has_pet ( stuid, petid )\n# pets ( petid, pet_age, pettype, weight )\n# has_pet.stuid = student.stuid\n# has_pet.petid = pets.petid\n\
    consistent: Yes\n\n\
    \
    Question: For all of the 4 cylinder cars, which model has the most horsepower?\n\
    SQL: SELECT model_list.Model FROM model_list JOIN car_makers ON model_list.Maker = car_makers.Id JOIN car_names ON model_list.ModelId = car_names.MakeId JOIN cars_data ON car_names.MakeId = cars_data.Id WHERE cars_data.Cylinders = 4 ORDER BY cars_data.Horsepower DESC LIMIT 1\n\
    database schema:\n\
    # cars_data ( horsepower, cylinders, id )\n# car_makers ( maker, id )\n# model_list ( model, maker )\n# car_names ( model, makeid )\n# model_list.maker = car_makers.id\n# car_names.model = model_list.model\n# cars_data.id = car_names.makeid\n\
    consistent: No\n\
    error cases: [2]\n\n\
    \
    Question: Which name is used by most number of students?\n\
    SQL: SELECT name FROM names JOIN used ON names.id = used.stuid  GROUP BY names.id ORDER BY count(*)\n\
    database schema:\n\
    # names ( id, first_name, player_id )\n# used (id, time) # students (id, name_id)# names.id = used.id\n# names.id = students.name_id\n\
    consistent: No\n\
    error cases: [3]\n\n\
    \
    Question: Which name is used by most number of students?\n\
    SQL: SELECT name FROM names JOIN used ON names.id = used.stuid  GROUP BY names.id,names.first_name ORDER BY count(*)\n\
    database schema:\n\
    # names ( id, first_name, player_id )\n# used (id, time) # students (id, name_id)# names.id = used.id\n# names.id = students.name_id\n\
    consistent: No\n\
    error cases: [1,3]\n\n\
    ## Let's begin. Please consider each case carefully and don't leave out any details. Remember to strictly follow the example format output, do not make any explanation:\n\
    Question: {question}\n\
    SQL: {sql}\n\
    database schema:\n\
    {db_schema}"
    PROMPT_REVISE = "## You are a text2sql expert. For a given SQL, I will provide you with a list of suggestions on revision, corresponding natural language question and database schema information.\
    Please follow the suggestions to revise the SQL one by one. \n\
    ## You must output the revised SQL directly without any explanation or prefix.\n\n\
    SQL: {sql}\n\
    Question: {question}\n\
    Database schema: {schema}\n\
    Suggestions for revision: \n\
    {suggestions}\
    Revised SQL: "
    PROMPT_REVISE_WITHOUT_SUGGESTION = "## You are an NL2SQL expert. For the given SQL and corresponding Question, I need you to check whether there are any errors in the SQL. \
    If yes, correct the original SQL and output the corrected SQL. Else, output the original SQL.\n\
    ## You must output the SQL directly without any explaination or prefix.\n\n\
    SQL: {sql}\n\
    Question: {question}\n\
    Database schema: {schema}\n\
    Corrected SQL: "

    suggestion_dict = {1: "If the ORDER BY keyword has multiple column conditions, keep only the most critical one.",
                       2: "For all WHERE clauses or JOIN conditions in an SQL query, make sure that the join between tables matches the correct primary-foreign key relationship.",
                       3: "Some tables that you need to answer this question do not appear in SQL. Find them from the schema information and put them into SQL correctly and ensure tables are correctly joined.",
                       4: "For the placeholder {placeholder} that appears in SQL, consider replacing it with the correct content in the context of natural language question.",
                       5: "Correct the use of schema refer to execution error message: [{error_message}]"}

    output_sqls = []
    num_question = max(d["question_index"] for d in datas) + 1
    print(num_question)
    for question_index in tqdm(range(num_question)):
        # 找出当前问题的全部候选sql
        candidates = [
            c for c in datas if c["question_index"] == question_index]
        # assert len(candidates) in [1,4]
        threshold = 0.9
        num_of_revise = 0
        # 开始循环，强行退出条件是循环到了3次
        while num_of_revise < 1:
            print(f"num_of_revise:{num_of_revise}")
            num_of_revise += 1
            # 1. verify环节
            # 首先用bert计算所有初步得分
            # key:sql_json
            # value:score
            candidates_bert_score = {}

            # key:sql_json
            # value:high_score
            high_candidates = {}
            for c in candidates:
                # print(c)
                exec_result = sql_execute(c['db_id'], c['final_sql'])
                bert_score = bert_verifier_for_score(
                    tokenizer, model, c['ori_question'], c['final_sql'], exec_result)
                candidates_bert_score[str(c)] = bert_score
                # print(bert_score)
                # 如果大于threshold就算高分
                if bert_score > threshold:
                    high_candidates[str(c)] = bert_score
            # print(candidates_bert_score.values())
            # 开始按高分数量分类处理
            # case1 只有1个高分，直接进入revision判定
            if len(high_candidates.keys()) == 1:
                candidates = [list(high_candidates.keys())[0]]
            # case2 大于1个高分
            elif len(high_candidates.keys()) > 1:
                # candidates = [max(candidates_bert_score, key=candidates_bert_score.get)]
                # key:exec_result
                # value:num
                tmp_exec = {}
                # 与tmp_exec一一对应
                # key:exec_result
                # value:sql_json
                tmp_candidate = {}
                # 遍历统计各个执行结果的出现频率
                for k in high_candidates.keys():
                    if isinstance(k, str):
                        k = eval(k)
                    exec_result = sql_execute(k['db_id'], k['final_sql'])
                    if exec_result[0] == '[':
                        exec_result = str(sorted(set(eval(exec_result))))
                    if exec_result not in tmp_exec.keys():
                        tmp_exec[exec_result] = 1
                        tmp_candidate[exec_result] = [k]
                    else:
                        tmp_exec[exec_result] += 1
                        tmp_candidate[exec_result].append(k)
                # 如果投票发现所有高分的执行结果都一致,直接进入revision判定
                if max(tmp_exec.values()) == len(high_candidates):
                    candidates = [list(high_candidates.keys())[0]]
                # 高分的也存在不一样的,那么所有高分的都进入revision判定
                else:
                    candidates = []
                    for h in list(high_candidates.keys()):
                        if isinstance(h, str):
                            h = eval(h)
                        candidates.append(h)
                    # 备用，只选zuigaofen
                    # max_score_c = max(high_candidates, key=high_candidates.get)
                    # if isinstance(max_score_c, str):
                    #     max_score_c = eval(max_score_c)
                    # candidates.append(max_score_c)
            # case3 没有高分的,那就强行选两个分较高的进入revision判定
            else:
                # candidates = [max(candidates_bert_score, key=candidates_bert_score.get)]
                candidates = sorted(
                    candidates_bert_score, key=candidates_bert_score.get, reverse=True)[:2]
                for c in candidates:
                    if isinstance(c, str):
                        c = eval(c)
                # print(candidates)

            # 2.revision环节
            tmp_score = 0
            tmp_final_candidate = None
            for c in candidates:
                # if candidates_bert_score[str(c)] > 0.9 and candidates_revision_cases[str(c)] == [] and candidates_bert_score[str(c)] > tmp_score:
                if candidates_bert_score[str(c)] > threshold and candidates_bert_score[str(c)] > tmp_score:
                    tmp_score = candidates_bert_score[str(c)]
                    tmp_final_candidate = c
            # 若存在符合条件的，那么无需revision，等待之后输出即可
            if tmp_final_candidate is not None:
                candidates = [tmp_final_candidate]
                # 唯一可以中途退出循环的条件，就是当前candidates中有分数足够高的
                break
            # 还是需要去修改
            else:
                # key：sql_json
                # value:error_cases
                candidates_revision_cases = {}
                for c in candidates:
                    # 2.1 判定是否需要revision，若需要，给出revision cases
                    if isinstance(c, str):
                        c = eval(c)
                    prompt = PROMPT_SELECT_CASE.format(
                        question=c['ori_question'], sql=c['final_sql'], db_schema=all_schemas[c['question_index']]['input_sequence'])
                    prompt = prompt.replace("    ", '')
                    # print(prompt)
                    token_count = len(encoding.encode(prompt))
                    total_tokens += token_count
                    n_repeat = 0
                    while True:
                        try:
                            res = chat_gpt(llm_model, prompt)
                            break
                        except Exception as e:
                            n_repeat += 1
                            print(
                                f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                            time.sleep(1)
                            continue
                    # print('result:'+str(res))
                    answer = res['response'][0]
                    token_count = len(encoding.encode(prompt))
                    total_response_tokens += token_count
                    try:
                        consistency = answer.split('error cases: ')[0]
                        if 'yes' in consistency.lower():
                            consistency = 'yes'
                            print(consistency)
                        else:
                            consistency = 'no'
                            error_cases = answer.split(
                                'error cases: ')[1].strip()
                            error_cases = eval(error_cases)
                            print(consistency)
                            print(error_cases)
                    except:
                        print("##############error\n")
                        consistency = 'no'
                        error_cases = [1, 2, 3, 4]
                    if consistency == 'yes':
                        candidates_revision_cases[str(c)] = []
                    else:
                        candidates_revision_cases[str(c)] = error_cases
                # 到这里得到了所有需要修改的candidates和error case
                print(f"candidates num : {len(candidates)}")
                for c in candidates:
                    print(candidates_bert_score[str(c)])
            ##################################################################
            # 2.2 根据判定结果进一步操作
            # 首先看存不存在已经可以输出了的，分数高且无语义差的candidate
            # tmp_score = 0
            # tmp_final_candidate = None
            # for c in candidates:
            #     # if candidates_bert_score[str(c)] > 0.9 and candidates_revision_cases[str(c)] == [] and candidates_bert_score[str(c)] > tmp_score:
            #     if candidates_bert_score[str(c)] > 0.8 and candidates_bert_score[str(c)] > tmp_score:
            #         tmp_score = candidates_bert_score[str(c)]
            #         tmp_final_candidate = c
            # # 若存在符合条件的，那么无需revision，等待之后输出即可
            # if tmp_final_candidate is not None:
            #     candidates = [tmp_final_candidate]
            #     # 唯一可以中途退出循环的条件，就是当前candidates中有分数高且语义一致的
            #     break
            # # 还是需要去修改
            # else:
            ######################################################################
                # 修改后的candidate装到这里
                new_candidates = []
                # 逐个修改
                for c in candidates:
                    if isinstance(c, str):
                        c = eval(c)
                    # 首先对4，5这两个case是否存在进行检查，补全error cases
                    question = c['ori_question']
                    sql = c['final_sql'].replace('\n', '')
                    error_cases = candidates_revision_cases[str(c)]
                    suggestion_str = ""
                    placeholder = ""
                    if 'value1' in sql:
                        placeholder = 'value1,'
                    if 'tb1' in sql:
                        placeholder += 'tb1,'
                    if 'tb2' in sql:
                        placeholder += 'tb2,'
                    if 'counter' in sql:
                        placeholder += 'counter,'
                    if placeholder != "" and placeholder[-1] == ',':
                        placeholder = placeholder[:-1]
                    if placeholder != "":
                        error_cases.append(4)
                    error_message = ""
                    tmp_exec_result = sql_execute(c['db_id'], c['final_sql'])
                    if tmp_exec_result[0] != '[':
                        error_message = tmp_exec_result.replace('\n', '')
                        error_cases.append(5)

                    # 然后根据error cases写好修改意见
                    for index_suggestion, suggestion in enumerate(error_cases):
                        str_index = str(index_suggestion+1)
                        if suggestion == 4:
                            suggestion_str += f"# {str_index}. {suggestion_dict[suggestion].format(placeholder=placeholder)}\n"
                        elif suggestion == 5:
                            suggestion_str += f"# {str_index}. {suggestion_dict[suggestion].format(error_message=error_message)}\n"
                        else:
                            suggestion_str += f"# {str_index}. {suggestion_dict[suggestion]}\n"

                    # 如果error cases为空，那不给建议直接修改
                    if error_cases == []:
                        prompt = PROMPT_REVISE_WITHOUT_SUGGESTION.format(
                            question=question, sql=sql, schema=all_schemas[c['question_index']]['input_sequence'])
                    else:
                        prompt = PROMPT_REVISE.format(
                            question=question, sql=sql, schema=all_schemas[c['question_index']]['input_sequence'], suggestions=suggestion_str)
                    prompt = prompt.replace("    ", "")
                    # print(prompt)
                    token_count = len(encoding.encode(prompt))
                    total_tokens += token_count
                    n_repeat = 0
                    while True:
                        try:
                            res = chat_gpt(llm_model, prompt)
                            break
                        except Exception as e:
                            n_repeat += 1
                            print(
                                f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                            time.sleep(1)
                            continue
                    # print('result:'+str(res))

                    revised_sql = res['response'][0]
                    token_count = len(encoding.encode(revised_sql))
                    total_response_tokens += token_count
                    revised_sql = revised_sql.replace("```sql", "")
                    revised_sql = revised_sql.replace("```", "").strip()
                    revised_sql = " ".join(
                        revised_sql.replace("\n", " ").split())
                    new_c = copy.deepcopy(c)
                    new_c['final_sql'] = revised_sql
                    new_candidates.append(new_c)
                candidates = new_candidates
        print('\n')
        if len(candidates) == 1:
            if isinstance(candidates[0], str):
                output_sqls.append(eval(candidates[0])['final_sql'])
            else:
                output_sqls.append(candidates[0]['final_sql'])
        else:
            final_bert_score = {}
            for c in candidates:
                bert_score = bert_verifier_for_score(
                    tokenizer, model, c['ori_question'], c['final_sql'], exec_result)
                final_bert_score[str(c)] = bert_score
            final_sql = max(final_bert_score, key=final_bert_score.get)
            if isinstance(final_sql, str):
                final_sql = eval(final_sql)
            output_sqls.append(final_sql['final_sql'])

        output_sql_path = root_path+"classified_revision.txt"
        with open(output_sql_path, 'w') as f:
            for i in output_sqls:
                f.write(i+"\n")
        print(total_response_tokens)
        print(total_tokens)


# 统计verify结果
def xyr_analysis_final(root_path):
    exec_result = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1]
    final_path = root_path+"8_final.json"
    ####
    with open(final_path, 'r') as f:
        data = json.load(f)
    f.close()
    for index, i in enumerate(data):
        i['verified_acc'] = exec_result[index]
        del i['query_toks']
        del i['question_toks']
        del i['query_toks_no_value']
        del i['sql']
    output_path = root_path + "9_new_verify_result.json"
    with open(output_path, 'w') as f:
        json.dump(data, f)
    f.close()


INPUT_PATH = "./extra_dev.json"
# INPUT_PATH = "extra_dev.json"
if __name__ == "__main__":
    # root_path = "output/9.27_spider_v2/"
    root_path = "output/10.13_experiments/llms/qwen/"
    # root_path = "output/5.23_sciencebenckmark/oncomx/"
    # root_path = "output/5.24_演示用/"

    # 1.问题分解，decomposition
    # input_path = "/Users/xianyiran/Desktop/实验室/text2sql/sciencebenchmark_dataset-master/oncomx/dev.json"
    # input_path = "extra_dev.json"
    input_path = INPUT_PATH
    model = "gpt-3.5-turbo"
    # model = "gpt-4o"
    temperature = 0.7

    # prompt_num = 0  # bridge-from
    # output_path = root_path+"1_bridge_from_decomposition_gpt3.5.json"
    # decomposition_question(input_path, output_path, model, prompt_num)
    # prompt_num = 1  # bridge-not
    # output_path = root_path+"1_bridge_not_decomposition_gpt3.5.json"
    # decomposition_question(input_path, output_path, model, prompt_num)
    # prompt_num = 2  # bridge-where
    # output_path = root_path+"1_bridge_where_decomposition_gpt3.5.json"
    # decomposition_question(input_path, output_path, model, prompt_num)
    # prompt_num = 3  # combination
    # output_path = root_path+"1_combination_decomposition_gpt3.5.json"
    # decomposition_question(input_path, output_path, model, prompt_num)

    # prompt_num = 4  # 2,3,3,2个shot，用于对比
    # output_path = "output/1_mix_decomposition_gpt3.5.json"
    # decomposition_question(input_path, output_path,
    #                        openai_key, model, temperature, prompt_num)
    # # 2.1.处理数据为dail可处理格式
    # decomposition_result_file = root_path+"1_bridge_from_decomposition_gpt3.5.json"
    # processed_file = root_path+"2.1_bridge_from_for_dail.json"
    # process_data_for_dail(decomposition_result_file, processed_file)

    # decomposition_result_file = root_path+"1_bridge_not_decomposition_gpt3.5.json"
    # processed_file = root_path+"2.1_bridge_not_for_dail.json"
    # process_data_for_dail(decomposition_result_file, processed_file)

    # decomposition_result_file = root_path+"1_bridge_where_decomposition_gpt3.5.json"
    # processed_file = root_path+"2.1_bridge_where_for_dail.json"
    # process_data_for_dail(decomposition_result_file, processed_file)

    # decomposition_result_file = root_path+"1_combination_decomposition_gpt3.5.json"
    # processed_file = root_path+"2.1_combination_for_dail.json"
    # process_data_for_dail(decomposition_result_file, processed_file)

    # decomposition_result_file = "output/1_mix_decomposition_gpt3.5.json"
    # processed_file = "output/2_mix_for_dail.json"
    # process_data_for_dail(decomposition_result_file, processed_file)

    # # 2.2.处理最后一个子问题为可处理格式
    # input_path = root_path + "1_bridge_from_decomposition_gpt3.5.json"
    # output_path = root_path + "2.2_bridge_from.json"
    # process_last_question(input_path, output_path)

    # input_path = root_path + "1_bridge_not_decomposition_gpt3.5.json"
    # output_path = root_path + "2.2_bridge_not.json"
    # process_last_question(input_path, output_path)

    # input_path = root_path + "1_bridge_where_decomposition_gpt3.5.json"
    # output_path = root_path + "2.2_bridge_where.json"
    # process_last_question(input_path, output_path)

    # input_path = root_path + "1_combination_decomposition_gpt3.5.json"
    # output_path = root_path + "2.2_combination.json"
    # process_last_question(input_path, output_path)

    # # 3.2 使用prompt生成最后一个子SQL
    # input_path = root_path + "2.2_bridge_from.json"
    # output_path = root_path + "3.2_bridge_from.txt"
    # last_sql_generate(input_path, output_path)

    # input_path = root_path + "2.2_bridge_not.json"
    # output_path = root_path + "3.2_bridge_not.txt"
    # last_sql_generate(input_path, output_path)

    # input_path = root_path + "2.2_bridge_where.json"
    # output_path = root_path + "3.2_bridge_where.txt"
    # last_sql_generate(input_path, output_path)

    # input_path = root_path + "2.2_combination.json"
    # output_path = root_path + "3.2_combination.txt"
    # last_sql_generate(input_path, output_path)

    # # 4.1.处理dail生成结果
    # processed_file = root_path+"2.1_bridge_from_for_dail.json"
    # dail_result = root_path+"3.1_bridge_from.txt"
    # output_processed_file = root_path+"4_bridge_from.json"
    # process_dail_result(processed_file, dail_result, output_processed_file)

    # processed_file = root_path+"2.1_bridge_where_for_dail.json"
    # dail_result = root_path+"3.1_bridge_where.txt"
    # output_processed_file = root_path+"4_bridge_where.json"
    # process_dail_result(processed_file, dail_result, output_processed_file)

    # processed_file = root_path+"2.1_bridge_not_for_dail.json"
    # dail_result = root_path+"3.1_bridge_not.txt"
    # output_processed_file = root_path+"4_bridge_not.json"
    # process_dail_result(processed_file, dail_result, output_processed_file)

    # processed_file = root_path+"2.1_combination_for_dail.json"
    # dail_result = root_path+"3.1_combination.txt"
    # output_processed_file = root_path+"4_combination.json"
    # process_dail_result(processed_file, dail_result, output_processed_file)

    # processed_file = "output/2_mix_for_dail.json"
    # dail_result = "output/3_mix.txt"
    # output_processed_file = "output/4_mix.json"
    # process_dail_result(processed_file, dail_result, output_processed_file)

    # # 4.2.处理最后一个子问题生成结果，和前面的子问题整理到一起
    # dail_file = root_path + "4_bridge_from.json"
    # lastsql_info = root_path + "2.2_bridge_from.json"
    # lastsql_result = root_path + "3.2_bridge_from.txt"
    # process_last_sql(dail_file, lastsql_info, lastsql_result)

    # dail_file = root_path + "4_bridge_not.json"
    # lastsql_info = root_path + "2.2_bridge_not.json"
    # lastsql_result = root_path + "3.2_bridge_not.txt"
    # process_last_sql(dail_file, lastsql_info, lastsql_result)

    # dail_file = root_path + "4_bridge_where.json"
    # lastsql_info = root_path + "2.2_bridge_where.json"
    # lastsql_result = root_path + "3.2_bridge_where.txt"
    # process_last_sql(dail_file, lastsql_info, lastsql_result)

    # dail_file = root_path + "4_combination.json"
    # lastsql_info = root_path + "2.2_combination.json"
    # lastsql_result = root_path + "3.2_combination.txt"
    # process_last_sql(dail_file, lastsql_info, lastsql_result)

    # # 5.使用gpt生成最终sql
    # processed_dail_result = root_path+"4_bridge_from.json"
    # output_path = root_path+"5_bridge_from.json"
    # composite_final_SQL(processed_dail_result, output_path)

    # processed_dail_result = root_path+"4_bridge_where.json"
    # output_path = root_path+"5_bridge_where.json"
    # composite_final_SQL(processed_dail_result, output_path)

    # processed_dail_result = root_path+"4_bridge_not.json"
    # output_path = root_path+"5_bridge_not.json"
    # composite_final_SQL(processed_dail_result, output_path)

    # processed_dail_result = root_path+"4_combination.json"
    # output_path = root_path+"5_combination.json"
    # composite_final_SQL(processed_dail_result, output_path)

    # processed_dail_result = "output/4_mix.json"
    # output_path = "output/5_mix.json"
    # composite_final_SQL(processed_dail_result, output_path)

    # # 6.生成供评估的gold与pre
    # final_result = root_path+"5_bridge_from.json"
    # eval_gold = root_path+"6_gold_bridge_from.txt"
    # eval_pred = root_path+"6_pred_bridge_from.txt"
    # process_data_for_eval(final_result, eval_gold, eval_pred)

    # final_result = root_path+"5_bridge_where.json"
    # eval_gold = root_path+"6_gold_bridge_where.txt"
    # eval_pred = root_path+"6_pred_bridge_where.txt"
    # process_data_for_eval(final_result, eval_gold, eval_pred)

    # final_result = root_path+"5_bridge_not.json"
    # eval_gold = root_path+"6_gold_bridge_not.txt"
    # eval_pred = root_path+"6_pred_bridge_not.txt"
    # process_data_for_eval(final_result, eval_gold, eval_pred)

    # final_result = root_path+"5_combination.json"
    # eval_gold = root_path+"6_gold_combination.txt"
    # eval_pred = root_path+"6_pred_combination.txt"
    # process_data_for_eval(final_result, eval_gold, eval_pred)

    # test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database",gold=root_path+"6_gold_bridge_from.txt",predict=root_path+"6_pred_bridge_from.txt",etype="exec",exec_result_path=root_path+"bridge_from_exec.txt")
    # test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database",gold=root_path+"6_gold_bridge_not.txt",predict=root_path+"6_pred_bridge_not.txt",etype="exec",exec_result_path=root_path+"bridge_not_exec.txt")
    # test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database",gold=root_path+"6_gold_bridge_where.txt",predict=root_path+"6_pred_bridge_where.txt",etype="exec",exec_result_path=root_path+"bridge_where_exec.txt")
    # test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database",gold=root_path+"6_gold_combination.txt",predict=root_path+"6_pred_combination.txt",etype="exec",exec_result_path=root_path+"combination_exec.txt")
    # test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database",gold=root_path+"gold_new_2000.txt",predict=root_path+"3_new_2000.txt",etype="exec",exec_result_path=root_path+"new_2000_exec.txt")
    # 7.把5加上执行结果
    # eval_score_process(root_path)

    # 8.统计正确率
    # analysis_acc(root_path)
    # lever_verify_acc(root_path)
    # xyr_verify_acc(root_path)
    # test_suite_evaluation(db_dir="/home3/xianyiran/text2sql/spider/database",gold=root_path+"6_gold_combination.txt",predict=root_path+"xyr_bert_verified_sql.txt",etype="exec",exec_result_path=root_path+"bert_verified_sql_exec.txt")
    # xyr_choose_correct_new(root_path)
    # #以下为sql2nl部分
    # xyr_sql2nl_verify(root_path)
    # xyr_revision(root_path)
    # xyr_revision_with_schema(root_path)
    # 最新的verify+revision
    xyr_classified_revision(root_path)

    # 9.分析verify结果
    # xyr_analysis_final(root_path)

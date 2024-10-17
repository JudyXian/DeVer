import json
import os
import re
import random
import sqlparse

from nltk.tokenize import word_tokenize
from nltk import ngrams
from sql_metadata import Parser
from pyserini.search.lucene import LuceneSearcher
from utils.bridge_content_encoder import get_matched_entries
from utils.db_utils import get_db_schema

import pandas as pd
from typing import Dict
from pathlib import Path

random.seed(42)


def load_tables_description(db_directory_path: str, use_value_description: bool) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Loads table descriptions from CSV files in the database directory.

    Args:
        db_directory_path (str): The path to the database directory.
        use_value_description (bool): Whether to include value descriptions.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary containing table descriptions.
    """
    encoding_types = ['utf-8-sig', 'cp1252']
    description_path = Path(db_directory_path) / "database_description"  #每个database 下面还有一个description文件  --->这个和evidence是不是可以互补?
    
    if not description_path.exists():
        print(f"Description path does not exist: {description_path}")
        return {}
    
    table_description = {}
    for csv_file in description_path.glob("*.csv"):
        table_name = csv_file.stem.lower().strip()
        table_description[table_name] = {}
        could_read = False
        for encoding_type in encoding_types:
            try:
                table_description_df = pd.read_csv(csv_file, index_col=False, encoding=encoding_type) #打开csv 描述文件
                for _, row in table_description_df.iterrows():
                    column_name = row['original_column_name']
                    expanded_column_name = row.get('column_name', '').strip() if pd.notna(row.get('column_name', '')) else ""
                    #关于column的描述信息
                    column_description = row.get('column_description', '').replace('\n', ' ').replace("commonsense evidence:", "").strip() if pd.notna(row.get('column_description', '')) else ""
                    #数据类型
                    data_format = row.get('data_format', '').strip() if pd.notna(row.get('data_format', '')) else ""
                    value_description = ""
                    #关于某一列 数据取值的描述信息
                    if use_value_description and pd.notna(row.get('value_description', '')):
                        value_description = row['value_description'].replace('\n', ' ').replace("commonsense evidence:", "").strip()
                        if value_description.lower().startswith("not useful"):
                            value_description = value_description[10:].strip()
                    
                    table_description[table_name][column_name.lower().strip()] = {
                        "original_column_name": column_name,
                        "column_name": expanded_column_name,
                        "column_description": column_description,
                        "data_format": data_format,
                        "value_description": value_description
                    }
                print(f"Loaded descriptions from {csv_file} with encoding {encoding_type}")
                could_read = True
                break
            except Exception as e:
                print(e)
                continue
    if not could_read:
        print(f"Could not read descriptions from {csv_file}")
    return table_description


def extract_large_numbers(text):
    number_information = []
    patterns = {
        'thousand': 10**3,
        'million': 10**6,
        'billion': 10**9,
        'trillion': 10**12
    }
    
    for word, multiplier in patterns.items():
        matches = re.findall(r'(\d+\.?\d*)\s*{}'.format(word), text, flags=re.IGNORECASE)
        for match in matches:
            number = float(match) * multiplier
            number_information.append(match + " " + word + " = " + str(int(number)))
    
    for phrase, number in {'thousands of': 10**3, 'millions of': 10**6, 'billions of': 10**9, 'trillions of': 10**12}.items():
        if phrase in text:
            number_information.append(phrase + " = " + str(int(number)))
    
    large_number_evidence = ""
    for info in number_information:
        large_number_evidence += info + "; "
    
    return large_number_evidence.strip()

def remove_table_alias(s):
    try:
        tables_aliases = Parser(s).tables_aliases
    except Exception as e:
        return s

    new_tables_aliases = {}
    for i in range(1,11):
        if "t{}".format(i) in tables_aliases.keys():
            new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
    
    tables_aliases = new_tables_aliases
    for k, v in tables_aliases.items():
        # remove AS clauses
        s = s.replace("AS " + k + " ", "")
        # replace table alias with thier original names
        s = s.replace(k, v)
    
    return s

def remove_similar_comments(names, comments):
    '''
    Remove table (or column) comments that have a high degree of similarity with their names
    
    Arguments:
        names: a list of table (or column) names
        comments: a list of table (or column) comments
    
    Returns:
        new_comments: a list of new table (or column) comments
    '''
    new_comments = []
    for name, comment in zip(names, comments):    
        # name.replace("_", "").replace(" ", "")
        # comment.replace("_", "").replace(" ", "")
        # new_comments.append(comment)
        if name.replace("_", "").replace(" ", "") == comment.replace("_", "").replace(" ", ""):
            new_comments.append("")
        else:
            new_comments.append(comment)
    
    return new_comments

def str_replace_ignore_case(evidence, schema_item_name):
    evidence = re.sub(re.escape(schema_item_name), schema_item_name, evidence, 0, re.IGNORECASE)

    return evidence

def obtain_n_grams(sequence, max_n):
    '''
    returns all grams of sequence less than or equal to `max_n`
    '''
    tokens = word_tokenize(sequence)
    all_grams = []
    for n in range(1, max_n + 1):
        all_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    
    return all_grams

def preprocess_evidence(evidence, schema_items):
    if evidence.strip() == "":
        return ""

    evidence = evidence.strip()
    # if evidence does not end with ";", add a ";" char
    if not evidence.endswith(";"):
        evidence += ";"
    
    # lowercase schema items appeared in the evidence
    for table in schema_items:
        if table["table_name"] in evidence.lower():
            evidence = str_replace_ignore_case(evidence, table["table_name"]) 

        for column_name in table["column_names"]:
            if column_name in evidence.lower():
                evidence = str_replace_ignore_case(evidence, column_name)
    
    evidence = evidence.replace("< =", "<=").replace("> =", ">=")

    return evidence

def spider_style_dataset(
    dataset_path, 
    db_path, 
    db_content_index_path, 
    source, 
    table_json_path,
    use_evidence,
    mode
):
    '''
    Load spider-style dataset
    
    dataset_path = os.path.join("/home3/xianyiran/text2sql/jzx_decompositionSQL/data/spider/", spider_train_set), 
    db_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/spider/database",  #数据库路径
    db_content_index_path = "./data/sft_data_collections/spider/db_contents_index",
    source = "spider-train",
    table_json_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/spider/tables.json",  #table schema路径
    use_evidence = False,
    mode = "train"

    Arguments:
        dataset_path: directory to load the dataset from
        db_path: directory of databases (used for extracting schema, including tables, columns, column contents, and foreign keys)
        db_content_index_path: directory of database content sparse index
        source: source of examples
        table_json_path: directory to load additional database information (used for extracting comments for tables and columns)
        use_evidence: whether to use the additional evidence in the input sequence
    Returns:
        returned_dataset: prepared dataset
    '''
    returned_dataset = []

    dataset = json.load(open(dataset_path)) #训练数据
    additional_db_info = json.load(open(table_json_path)) #schema 信息

    db_comments = dict()

    
    # for table_name, columns in table_description.items():
    #     for column_name, column_info in columns.items():
    #         metadata = {
    #             "table_name": table_name,
    #             "original_column_name": column_name,
    #             "column_name": column_info.get('column_name', ''),
    #             "column_description": column_info.get('column_description', ''),
    #             "value_description": column_info.get('value_description', '') if kwargs.get("use_value_description", True) else ""
    #         }


    # record comments for tables and columns
    for db_info in additional_db_info:
        db_directory_path = db_path+'/'+db_info['db_id']
        table_description = load_tables_description(db_directory_path, True)  #每个database 的描述信息

        comment_dict = dict()
        #把 column names 加进去   这时候应该也要把description信息加进来
        column_names = [column_name.lower() for _, column_name in db_info["column_names_original"]] #列名
        table_idx_of_each_column = [t_idx for t_idx, _ in db_info["column_names_original"]] #列对应的表名id
        column_comments = [column_comment.lower() for _, column_comment in db_info["column_names"]] #每个列的 描述信息   还需要加列的description信息
        
        assert len(column_names) == len(column_comments)
        # column_comments = remove_similar_comments(column_names, column_comments)  #去掉 column_name 与 column_name_original 相同的数据; 相同的数据将column_comments置为空

        table_names = [table_name.lower() for table_name in db_info["table_names_original"]] #表名
        table_comments = [table_comment.lower() for table_comment in db_info["table_names"]]
        
        assert len(table_names) == len(table_comments)
        # table_comments = remove_similar_comments(table_names, table_comments) #去掉表名相同的部分   相同部分将table_comments置为空

        # enumerate each table and its columns
        for table_idx, (table_name, table_comment) in enumerate(zip(table_names, table_comments)):
            comment_dict[table_name] = {
                "table_comment": table_comment,
                "column_comments": dict(),
                "column_description": dict()
            }
            for t_idx, column_name, column_comment in zip(table_idx_of_each_column, column_names, column_comments):
                # record columns in current table
                if t_idx == table_idx: #对应表的列
                    comment_dict[table_name]["column_comments"][column_name.lower().strip()] = column_comment
                    #加入每一列的  description信息
                    print(f"db: {db_info['db_id']}  table_name: {table_name}  column_comment: {column_comment}   column_name: {column_name}")
                    if column_comment.lower().strip() in table_description[table_comment].keys():
                        comment_dict[table_name]["column_description"][column_name.lower().strip()] = {
                            'column_description':table_description[table_comment][column_comment.lower().strip()]['column_description'], 
                            'value_description':table_description[table_comment][column_comment.lower().strip()]['value_description']}
                    elif column_name.lower().strip() in table_description[table_comment].keys():
                        comment_dict[table_name]["column_description"][column_name.lower().strip()] = {
                            'column_description':table_description[table_comment][column_name.lower().strip()]['column_description'], 
                            'value_description':table_description[table_comment][column_name.lower().strip()]['value_description']}
                    else:
                        # print(table_description[table_comment].keys())
                        # print('error')
                        # exit(-1)
                        continue

        db_comments[db_info["db_id"]] = comment_dict

    db_ids = set([data["db_id"] for data in dataset]) #所有数据库的名字
    db_id2searcher = dict()
    for db_id in db_ids:
        db_id2searcher[db_id] = LuceneSearcher(os.path.join(db_content_index_path, db_id)) #打开对应数据的BM25 index

    db_id2schema = dict()

    for data in dataset:
        sample = {}
        db_id = data["db_id"]
        
        sample["db_id"] = db_id
        sample["db_path"] = os.path.join(db_path, db_id, db_id + ".sqlite")

        if db_id in db_id2schema:
            sample["schema"] = db_id2schema[db_id]
        else:
            db_id2schema[db_id] = get_db_schema(sample["db_path"], db_comments, db_id)  #获取与该问题相关的table schema信息
            sample["schema"] = db_id2schema[db_id]

        if "spider-syn" in source:
            sample["question"] = data["SpiderSynQuestion"]
            sample["evidence"] = ""
        elif "bird" in source:
            sample["question"] = data["question"]
            evidence = preprocess_evidence(data["evidence"], sample["schema"]["schema_items"]) #bird数据集会有evidence   这个预处理没啥用
            sample["evidence"] = evidence
        elif "bank" in source:
            sample["question"] = data["question"]
            sample["evidence"] = extract_large_numbers(data["question"])
        else:
            sample["question"] = data["question"]
            sample["evidence"] = ""
        
        if "\n" in sample["question"]:
            sample["question"] = sample["question"].replace("\n", " ")
        if "\n" in sample["evidence"]:
            sample["evidence"] = sample["evidence"].replace("\n", " ")
        
        sample["text"] = sample["evidence"] + " " + sample["question"] \
            if use_evidence and sample["evidence"] != "" else sample["question"]  #规范格式 text就是question

        if mode in ["train", "dev"]:
            sql = data["SQL"] if source in ["bird-dev", "bird-train"] else data["query"]
            sample["sql"] = remove_table_alias(sqlparse.format(sql, keyword_case = "upper", identifier_case = "lower"))  #删除sql中的别名
        elif mode == "test":
            sample["sql"] = ""
        
        sample["table_labels"], sample["column_labels"] = [], []  #标记table / column name是否出现在 SQL中
        try:
            sql_tokens = [token.value for token in Parser(sample["sql"].lower()).tokens]  #sql解析成tokens
        except Exception as e:
            sql_tokens = sample["sql"].lower().split()
        
        for table_info in sample["schema"]["schema_items"]:
            if mode in ["train", "dev"]:
                table_name = table_info["table_name"]
                # 存在哪些 table/column name 就把label标为1
                sample["table_labels"].append(1 if table_name in sql_tokens else 0)
                sample["column_labels"].append([1 if column_name in sql_tokens or table_name+"."+column_name in sql_tokens else 0 \
                    for column_name in table_info["column_names"]])
            elif mode == "test":
                sample["table_labels"].append(0)  #所有label全为0
                sample["column_labels"].append([0 for _ in range(len(table_info["column_names"]))])

        # coarse-grained matching between the input text and all contents in database
        # 与数据库内容的匹配 首先是粗粒度的匹配  使用BM25 匹配问题中的字符串
        grams = obtain_n_grams(sample["text"], 4)
        hits = []
        searcher = db_id2searcher[db_id]  #db_id2searcher: BM25索引
        for query in grams:
            hits.extend(searcher.search(query, k = 10))
        
        # hits = searcher.search(sample["text"], k = 50)

        coarse_matched_contents = dict()
        for i in range(len(hits)):
            matched_result = json.loads(hits[i].raw)
            # `tc_name` refers to column names like `table_name.column_name`, e.g., document_drafts.document_id
            # tc_name 匹配的列名
            tc_name = ".".join(matched_result["id"].split("-**-")[:2])
            if tc_name in coarse_matched_contents.keys():
                if matched_result["contents"] not in coarse_matched_contents[tc_name]:
                    coarse_matched_contents[tc_name].append(matched_result["contents"])
            else:
                coarse_matched_contents[tc_name] = [matched_result["contents"]]
        
        fine_matched_contents = dict()
        for tc_name, contents in coarse_matched_contents.items(): #细粒度的匹配  转成embedding 比较相似度?
            # fine-grained matching between the question and coarse matched contents
            fm_contents = get_matched_entries(sample["text"], contents)
            
            if fm_contents is None:
                continue
            for _match_str, (field_value, _s_match_str, match_score, s_match_score, _match_size,) in fm_contents:
                if match_score < 0.9:
                    continue
                if tc_name in fine_matched_contents.keys():
                    if len(fine_matched_contents[tc_name]) < 25:
                        fine_matched_contents[tc_name].append(field_value.strip())
                else:
                    fine_matched_contents[tc_name] = [field_value.strip()]

        sample["matched_contents"] = fine_matched_contents  #匹配结果  --> 匹配的value 应该和对应的table  column name 结合起来
        sample["source"] = source

        returned_dataset.append(sample)

    del db_id2searcher

    return returned_dataset

#生成符合train_schema_item_filter格式的数据 
if __name__ == "__main__":
    print("preparing training sets.....")
    print("spider-train")
    spider_train = []
    # Spider training set-1 (7000 + 1658 examples)
    # for spider_train_set in ["train_spider.json", "train_others.json"]:
    #     spider_train.extend(
    #         spider_style_dataset(
    #             dataset_path = os.path.join("/home3/xianyiran/text2sql/jzx_decompositionSQL/data/spider/", spider_train_set), 
    #             db_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/spider/database",  #数据库路径
    #             db_content_index_path = "./data/sft_data_collections/spider/db_contents_index",
    #             source = "spider-train",
    #             table_json_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/spider/tables.json",  #table schema路径
    #             use_evidence = False,
    #             mode = "train"
    #         )
    #     )
    # with open("./data/sft_spider_train_text2sql.json", "w") as f:
    #     f.write(json.dumps(spider_train, indent = 2, ensure_ascii = False))

    # print("BIRD (without evidence) train")
    # # BIRD training set (9428 examples)
    # bird_train = spider_style_dataset(
    #     dataset_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/bird/train/train.json", 
    #     db_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/bird/train/train_databases",  #数据库路径
    #     db_content_index_path = "./data/sft_data_collections/bird/train/db_contents_index",
    #     source = "bird-train",
    #     table_json_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/bird/train/train_tables.json",
    #     use_evidence = False,
    #     mode = "train"
    # )
    # with open("./data/sft_bird_train_text2sql.json", "w") as f:
    #     f.write(json.dumps(bird_train, indent = 2, ensure_ascii = False))

    print("BIRD (with evidence) train")
    # BIRD training set with evidence (9428 examples)
    bird_with_evidence_train = spider_style_dataset(
        dataset_path = "./data/bird/train/train.json", 
        db_path = "./data/bird/train/train_databases", 
        db_content_index_path = "./data/sft_data_collections/bird/train/db_contents_index",
        source = "bird-train",
        table_json_path = "./data/bird/train/train_tables.json",
        use_evidence = True,
        mode = "train"
    )
    with open("./data/sft_bird_with_evidence_train_text2sql_new.json", "w") as f:
        f.write(json.dumps(bird_with_evidence_train, indent = 2, ensure_ascii = False))


    print("BIRD-dev (with evidence)")
    # BIRD dev set (1534 examples)
    bird_with_evidence_dev = spider_style_dataset(
        dataset_path = "./data/bird/dev/dev.json", 
        db_path = "./data/bird/dev/dev_databases", 
        db_content_index_path = "./data/sft_data_collections/bird/dev/db_contents_index",
        source = "bird-dev",
        table_json_path = "./data/bird/dev/dev_tables.json",
        use_evidence = True,
        mode = "dev"
    )
    with open("./data/sft_bird_with_evidence_dev_text2sql_new.json", "w") as f:
        f.write(json.dumps(bird_with_evidence_dev, indent = 2, ensure_ascii = False))
    
    # print("Spider + BIRD train set (ALL MERGED)")
    # # merge all available training data
    # with open("./data/sft_all_merged_train_text2sql.json", "w") as f:
    #     f.write(json.dumps(spider_train + bird_with_evidence_train, indent = 2, ensure_ascii = False))
    
    # print("---------------------------------------------------------------------------")
    # print("preparing dev sets.....")
    # print("DR.spider")
    # dr_spider = []
    # # Dr.Spider has 17 perturbation test sets
    # test_set_names = os.listdir("/home3/xianyiran/text2sql/jzx_decompositionSQL/data/drspider/diagnostic-robustness-text-to-sql/data")
    # test_set_names.remove("Spider-dev")
    # for test_set_name in test_set_names:
    #     if test_set_name.startswith("DB_"):
    #         database_file_path = "database_post_perturbation"
    #         table_file_name = "tables_post_perturbation.json"
    #     else:
    #         database_file_path = "databases"
    #         table_file_name = "tables.json"
    #     dr_spider.extend(
    #             spider_style_dataset(
    #             dataset_path = os.path.join("/home3/xianyiran/text2sql/jzx_decompositionSQL/data/drspider/diagnostic-robustness-text-to-sql/data/", test_set_name, "questions_post_perturbation.json"), 
    #             db_path = os.path.join("/home3/xianyiran/text2sql/jzx_decompositionSQL/data/drspider/diagnostic-robustness-text-to-sql/data/", test_set_name, database_file_path), 
    #             db_content_index_path = os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "db_contents_index"),
    #             source = "dr.spider-{}".format(test_set_name),
    #             table_json_path = os.path.join("/home3/xianyiran/text2sql/jzx_decompositionSQL/data/drspider/diagnostic-robustness-text-to-sql/data/", test_set_name, table_file_name),
    #             use_evidence = False,
    #             mode = "dev"
    #         )
    #     )
    # with open("./data/sft_dr_spider_text2sql.json", "w") as f:
    #     f.write(json.dumps(dr_spider, indent = 2, ensure_ascii = False))
    
    # print("spider-dev")
    # # Spider development set (1034 examples)   验证集   测试 schema linking  需要改变下列的数据集相关路径
    # spider_dev = spider_style_dataset(
    #     dataset_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/spider/dev.json", 
    #     db_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/spider/database", 
    #     db_content_index_path = "./data/sft_data_collections/spider/db_contents_index",
    #     source = "spider-dev",
    #     table_json_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/spider/tables.json",
    #     use_evidence = False,
    #     mode = "dev"
    # )
    # with open("./data/sft_spider_dev_text2sql.json", "w") as f:
    #     f.write(json.dumps(spider_dev, indent = 2, ensure_ascii = False))

    # print("BIRD-dev (without evidence)")
    # # BIRD dev set (1534 examples)
    # bird_dev = spider_style_dataset(
    #     dataset_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/bird/dev/dev.json", 
    #     db_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/bird/dev/dev_databases", 
    #     db_content_index_path = "./data/sft_data_collections/bird/dev/db_contents_index",
    #     source = "bird-dev",
    #     table_json_path = "/home3/xianyiran/text2sql/jzx_decompositionSQL/data/bird/dev/dev_tables.json",
    #     use_evidence = False,
    #     mode = "dev"
    # )
    # with open("./data/sft_bird_dev_text2sql.json", "w") as f:
    #     f.write(json.dumps(bird_dev, indent = 2, ensure_ascii = False))

    
    
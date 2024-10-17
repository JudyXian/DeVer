from .schema_filter import filter_func, SchemaItemClassifierInference
import json
# in the eval mode, you do not need to provide sql,
# the relevant scores of tables and columns are predicted by the fine-tuned schema filter model based on the user's text (or question)


def schema_filter(input_path, output_path):
    with open(input_path, 'r') as f:
        dataset = json.load(f)

    # remain up to 3 relavant tables in the database
    num_top_k_tables = 3
    # remain up to 3 relavant columns for each remained table
    num_top_k_columns = 3

    # load fine-tuned schema filter
    sic = SchemaItemClassifierInference("/home3/xianyiran/text2sql/decomposition_and_DAIL/src/schema_filter/sic_merged")

    result = filter_func(
        dataset=dataset,
        dataset_type="eval",
        sic=sic,
        num_top_k_tables=num_top_k_tables,
        num_top_k_columns=num_top_k_columns
    )
    with open(output_path, 'w') as f:
        json.dump(result, f)


# input_path = "./sft_bird_dev_text2sql.json"
# output_path = "./schema_6_6.json"

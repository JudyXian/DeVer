import json
import argparse


def parse_option():
    parser = argparse.ArgumentParser(
        "command line arguments for generate prompt")
    parser.add_argument("--input_dataset_path", type=str)
    parser.add_argument("--output_dataset_path", type=str)

    opt = parser.parse_args()

    return opt


def c3_prompt_generate(input_dataset_path,output_dataset_path):
    with open(input_dataset_path) as f:
        data_all = json.load(f)
    temp = []
    for id, data in enumerate(data_all):
        # xyr为了获取schema信息更改
        data['input_sequence'] = ""
        # data['input_sequence'] = "### Complete sqlite SQL query only and with no explanation, and do not select extra columns that are not explicitly requested in the query. " \
        #                 "\n ### Sqlite SQL tables, with their properties: \n#\n"
        schema = ""
        for tab, cols in data['schema'].items():
            schema += '# ' + tab + ' ( '
            for i, col in enumerate(cols):
                schema += col
                if data['db_contents'][tab][i]:
                    schema += '("'
                    for value in data['db_contents'][tab][i]:
                        schema += value + '", "'
                    schema = schema[:-4] + '")'
                schema += ', '
            schema = schema[:-2] + ' )\n'
        data['input_sequence'] += schema[:-1]
        for fk in data['fk']:
            data['input_sequence'] += '\n# ' + fk
        # xyr为了打印schema信息注释掉
        # data['input_sequence'] += '\n#\n### ' + data['question'] + '\nSELECT'
    with open(output_dataset_path, 'w') as f:
        json.dump(data_all, f, indent=2)

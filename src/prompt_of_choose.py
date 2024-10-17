import json
import openai
import time
import copy

#jzx TODO: 1. BIRD数据集更换对应的例子    2. 加入schema 信息
PROMPT = "## You are a question decomposition master. I have four types of problem decomposition templates, \
respectively bridge-from, bridge-where, bridge-not and combination.\
your task is to determine whether a question needs to be decomposed, and if so, select one template for the question decomposition \
from these four types of templates for the given natural language problem. \n\
## Below I will give some examples of problems that are suitable for decomposition using each of the four types of templates:\n\n\
\
Question: Show the status of the city that has hosted the greatest number of competitions.\n\
Need decompose: Yes\n\
Template_type: bridge-from\n\n\
\
Question: List the states and number of invoices in the US.\n\
Need decompose: No\n\
\
Question: Show all locations and the number of gas stations in each location ordered by the count.\n\
Need decompose: Yes\n\
Template_type: bridge-from\n\n\
\
Question: Return the effective date of the claim that has the largest total settlement amount in 'tb1'.\n\
Need decompose: No\n\
\
Question: What are the first and last names of all the employees and how many people report to them?\n\
Need decompose: Yes\n\
Template_type: bridge-from\n\n\
\
Question: What is the average bike availability in stations that are not located in Palo Alto?\n\
Need decompose: Yes\n\
Template_type: bridge-not\n\n\
\
Question: List the first and last name of the students who do not have any food type allergy.\n\
Need decompose: Yes\n\
Template_type: bridge-not\n\n\
\
Question: What is the average bike availabiliy in stations which not in 'tb1'?\n\
Need decompose: No\n\
\
Question: Find the average rating star for each movie that not in 'tb1'.\n\
Need decompose: No\n\
\
Question: On which day and in which zip code was the min dew point lower than any day in zip code 94107?\n\
Need decompose: Yes\n\
Template_type: bridge-where\n\n\
\
Question: On which day and in which zip code the min dew point was less than 'value1'?\n\
Need decompose: No\n\
\
Question: Return the effective date of the claim that has the largest total settlement amount in 'tb1'.\n\
Need decompose: Yes\n\
Template_type: bridge-where\n\n\
\
Question: Return the average price of all products.\n\
Need decompose: No\n\
\
Question: Find the team names of the universities whose enrollments are smaller than the average enrollment size.\n\
Need decompose: Yes\n\
Template_type: bridge-where\n\n\
\
Question: What are the different models created by either the car maker General Motors or weighed more than 3500?\n\
Need decompose: Yes\n\
Template_type: combination\n\n\
\
Question: List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.\n\
Need decompose: Yes\n\
Template_type: combination\n\n\
\
Question: List the states both in 'tb1' and 'tb2'.\n\
Need decompose: No\n\
\
Question: What are the names of all stations that in 'tb1' but not in 'tb2'?\n\
Need decompose: No\n\n\
\
## Don't explain, just give an answer in an example format.\n\
## Let's begin:\n\n"


def chat_gpt(model, prompt):
    openai.api_key ='sk-xx'
    response = openai.ChatCompletion.create(
        model=model,
        # prompt=prompt,
        messages=[{"role": "user", "content": prompt}],
        stop=[";"]
    )
    response_clean = [choice["message"]["content"]
                      for choice in response["choices"]]
    return dict(
        response=response_clean,
        **response["usage"]
    )


def choose_prompt(input_path, output_path,  model):
    with open(input_path, 'r') as f: 
        data = json.load(f)
    output_list = []
    index = 0

    for one_data in data:
        print('type choose index:'+str(index))
        index += 1

        prompt = PROMPT + "Question: " + \
            one_data['question'] + "\n"  # orignal question
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

        print('result:'+str(res))
        choose_information = res['response'][0]
        # one_data.clear()
        try:
            choose_information = choose_information.replace('\n','')
            answer_list = choose_information.split('Template_type: ')
            # 说明没有template type，即不用分解
            if len(answer_list)==1:
                one_data['need decompose'] = "no"
                one_data['node_index'] = -1
            # 不然就需要分解
            elif len(answer_list)==2:
                one_data['type'] = answer_list[1]
            # 
            else:
                assert 1==0
        except:
            print("error:   "+choose_information)
            one_data['type'] = []
        output_list.append(res)
    with open(output_path, 'w') as f:
        # json.dump(output_list, f, indent=6)
        json.dump(data, f, indent=6)
    return output_list


def choose_prompt_2(input_path, output_path,  model):
    with open(input_path, 'r') as f:
        datas = json.load(f)
    output_list = []
    index = 0
    for data in datas:
        print('type choose index:'+str(index))
        index += 1
        # 上一轮就没分解过，那这一轮直接pass
        if "need decompose" in data.keys() and data['need decompose'] == "no":
            data_new = copy.deepcopy(data)
            # del data_new['interaction_pred']
            data_new['parent_question'] = data['question']
            # data_new['node_index'] = -1
            output_list.append(data_new)
        # 否则都需要判断分解类型
        else:
            for data_index, one_data in enumerate(data['interaction_pred']):
                prompt = PROMPT + "Question: " + \
                    one_data['question'] + "\n"  # orignal question
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

                print('result:'+str(res))
                choose_information = res['response'][0]
                new_data = copy.deepcopy(data)
                del new_data['interaction_pred']
                new_data['parent_type'] = data['decompose_type']
                del new_data['decompose_type']
                if "need decompose" in data.keys():
                    del new_data['need decompose']
                new_data['parent_question'] = one_data['question']
                new_data['node_index'] = one_data['node_index']
                try:
                    choose_information = choose_information.replace('\n','')
                    answer_list = choose_information.split('Template_type: ')
                    # 说明没有template type，即不用分解
                    if len(answer_list)==1:
                        new_data['need decompose'] = "no"
                    # 不然就需要分解
                    elif len(answer_list)==2:
                        new_data['type'] = answer_list[1]
                        new_data['need decompose'] = "yes"
                    # 
                    else:
                        assert 1==0
                except:
                    print("error:   "+choose_information)
                    new_data['type'] = []
                output_list.append(new_data)
        with open(output_path, 'w') as f:
            json.dump(output_list, f, indent=6)
    return output_list


# if __name__ == "__main__":
    # input_path = "/Users/xianyiran/Desktop/实验室/text2sql/decomposition_and_DAIL/output/7.23_BIRD_v3/ratio_dev_200.json"
    # output_path = "/Users/xianyiran/Desktop/实验室/text2sql/decomposition_and_DAIL/output/7.23_BIRD_v3/0.0_type_gpt4o.json"
    # model = "gpt-4o"
    # choose_prompt(input_path, output_path,  model)

    # input_path = "/Users/xianyiran/Desktop/实验室/text2sql/decomposition_and_DAIL/output/7.23_BIRD_v3/0.1_decomposition_gpt4o.json"
    # output_path = "/Users/xianyiran/Desktop/实验室/text2sql/decomposition_and_DAIL/output/7.23_BIRD_v3/1.0_type_gpt4o.json"
    # model = "gpt-4o"
    # choose_prompt_2(input_path, output_path,  model)

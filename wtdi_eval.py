import os
import random
import shutil
from io import StringIO
import time
import math
import re
import secrets
import string
import argparse
import hashlib # temp
from tqdm import tqdm

from openai import OpenAI
import pandas as pd

import utils

parser = argparse.ArgumentParser()

# Seed Option
parser.add_argument('-seed', default='-1', type=str)

# Output Option
parser.add_argument('-overwrite', default=0, choices=[0, 1], type=int)

# Mode Option
parser.add_argument('-mode', default='eval', type=str)
args = parser.parse_args()

tqdm.pandas()

API_KEY = ''

CLIENT = OpenAI(api_key=API_KEY)
RESOURCE_DIR = './resources'

def image_eval(seed, batch_num, image_files):
    file_id_dict = utils.upload_file(CLIENT, seed, batch_num, image_files)
    task_prompt = f'내가 제공한 New York Times restaurant review에 나온 사진들을 보면 각 사진에 여러 가지 정보들이 사실 있어. 그 정보의 category를 나눈다면 food, interior, exterior, customer, staff, chef, kitchen이 있어. 이제 각 사진의 각 category마다 그 category가 사진을 보는 뉴욕시나 그 근교의 소비자로 하여금 얼마나 그 식당에서 dine-in하고 싶게 하는지 (willingness to dine in)에 대하여 평가해서 0-1사이의 점수로 0.01 단위로 측정하고 싶어. 평가에는 다른 정보를 이용하면 편향이 생기니 내가 여기서 설명한 정보와 사진만을 이용해서 평가해줘. 그리고 앞에서 내가 너와 대화한 내용의 history는 지금 답변을 하는 것에 이용하지 말아줘.\n\
    그런데 category가 많으니, category를 줄여서 다음과 같은 4가지 카테고리에 대하여 willingness to dine in을 알려줬으면 해. 구체적으로, \n\
    1. image_filename – 각 사진은 내가 다음과 같이 준 순서대로 이름을 넣어줘: {', '.join(list(file_id_dict.keys()))}  \n\
    각 사진에 대해 해당 카테고리에 관련된 내용이 얼마나 되는지 그 proportion을 채워주면 돼. 각 사진마다 카테고리들의 합은 당연히 1이 되어야 해. 0.05점 단위로 평가하는 경향성이 있던데 그러지 말고 0.01단위로 정밀하게 평가해줘. 단, 음식이 포커싱되고 다른 카테고리가 거의 보이지 않는 경우에는 음식만 있는 것으로 간주해. \n\
    2. prop_food\n\
    3. prop_place \n\
    4. prop_customer \n\
    5. prop_chef_staff \n\
    6. wtdi_food: 사진의 food 이미지가 주는 willingness to dine in. food가 잘 안보이거나 일부만 보이거나 전체 사진에서 차지하는 부분이 크지 않으면 평가하지말고 이럴 경우 none이라고 표시해줘. 즉, 음식이 명확하게 잘 보일 때 평가해줘. \n\
    7. wtdi_place: 사진의 place (interior, exterior, kitchen) 이미지가 주는 willingness to dine in. 다만, food를 담은 plate는 place 이미지로 판단해서 평가하지마. 그리고 place 이미지 중에 벽이 없는 이미지는 평가하지마 마. 또한, 사진이 주로 food를 크게 보여주는 사진에서 place는 평가하지 말아줘. \n\
    8. wtdi_customer: 사진 속의 customer들과 그들이 식사하는 모습에 관한 이미지가 주는 willingness to dine in \n\
    9. wtdi_chef_staff: 사진 속의 chef와 staff에 대한 이미지가 주는 willingness to dine in\n\
    10. price_estimate: 위의 평가와 함께 각 사진의 정보를 보고 이 식당의 음식 가격대가 어느 정도일지 추정해줘. 만약 음식만 있는 경우 (prop_food = 1이고 나머지 prop_* = 0인 경우), 절대 사전 정보를 이용해서 불필요한 요인으로 가격대를 높게 평가하지마. 예를 들어, 사진으로 알 수 없는 정보인 고급 코스에 포함되는 요리인지로 가격대를 추정하지마. 무조건 사진에 보이는 음식 그 자체로만 평가해. 만약 음식이 보이지 않는 경우 (prop_food = 0이고 wtdi_from_food가 none) 나머지 카테고리의 wtdi로만 음식 가격대가 어느정도일지 추정해줘. 제일 중요한 점: 가격대는 1-5 사이의 값으로 0.01 단위로 평가해. 0.1단위로 좀 평가하지말고. 1이 가장 싼 곳이고 5가 가장 비싼 곳을 의미하게 표시해줘.\n\
    willingess to dine in 각각에 대해 평가된 점수에 대해서는 그렇게 평가한 이유에 대해서도 설명해줘. 만약 none 값을 가지는 경우 아래 칼럼들도 당연히 none이어야해. \n\
    11. wtdi_food_reason\n\
    12. wtdi_place_reason\n\
    13. wtdi_customer_reason\n\
    14. wtdi_chef_staff_reason\n\
    각 사진에 대하여 위에서 말한 것들의 각각에 대한 정보가 없어서 평가가 불가능하거나 내가 평가하지 말라고 한 것은 none이라고 표시해줘. 영어로 써. 각 none으로 평가되지 않은 카테고리는 세미콜론(;)으로 구분해서 이유 적어줘.\n\
    15. price_component: 가격을 왜 그렇게 추정헀는지 요인별로 분해해서 적어. 각 요인별 분해된 점수는 추정한 가격 점수와 동일해야해.\n\
    각 사진에 대하여 다음과 같은 사진 특성들을 알려줘. 0.01단위로, 0과 1사이의 수치로 알려줘야해.\n\
    16.	brightness\n\
	17. colorfulness\n\
    18. color_complexity\n\
	19. focus\n\
	20. depth_of_field\n\
	21. contrast\n\
	22. saturation\n\
	23. sharpness\n\
	24. vibrancy\n\
    사진에 보이는 고객 수가 몇 명인지 추정해줘. 만약 prop_customer 값이 0이고 wtdi_customer 값이 none이면 0으로 처리해주고. \n\
    25. num_cust \n\
    사진의 공간 크기도 추정해줘. 2차원 단위로 cm^2과 3차원 단위로 cm^3 각각으로. \n\
    26. space_size_2d \n\
    27. space_size_3d \n\
    마지막으로, 모든 것을 다 고려해서 소비자가 느낄 overall willingness to dine in으로 quality를 0.01단위로, 0과 1 사이의 수치로 측정해줘. \n\
    28. overall_wtdi \n\
    당연하겠지만 2번 ~ 5번 각 항목이 0이면, 대응하는 6번 ~ 9번 wtdi 칼럼들도 none이어야해. 거꾸로, 6번 ~ 9번 wtdi 칼럼이 none이면, 대응하는 2번 ~ 5번 항목들도 0이어야 해. 그 규칙이 맞는지도 점검하고.\n\
    위 번호로 나열한 항목될 외 너가 임의로 칼럼을 넣지마. CSV 형식으로 요약해주고 1번, 11번, 12번 칼럼은 값을 큰 따옴표(")로 감싸. 테이블 외 어떠한 추가적인 코멘트도 넣지마.'
    input_list = []
    input_list.append({"type": "input_text", "text": task_prompt})
    for file_id in file_id_dict.values():
        input_list.append({
            "type": "input_image",
            "file_id": file_id
            }
        )

    response = CLIENT.responses.create(
        model="gpt-5",
        input=[
        {
            "role": "user",
            "content": input_list
        }],
    )
    input_token = response.usage.input_tokens * 2.5 / 1000000
    output_token = response.usage.output_tokens * 2.5 / 1000000
    price = input_token + output_token
    try:
        df = pd.read_csv(StringIO(response.output_text), quotechar='"')
        unique_path, filename = utils.get_unique_filename(output_dir, batch_num, prefix='batch_', ext='.csv')
        df.to_csv(unique_path, index=False)
        print(f'[Seed #{seed}][Batch #{batch_num}] Done. Cost: ${price}')
        return True, filename
    except Exception as e:
        print(f'[Seed #{seed}][Batch #{batch_num}] Failed to obtain the result: {e}')
        return False, None

def df_check(df: pd.DataFrame, file_name: str, batch_dict: dict):
    batch_num = int(file_name.split('_')[1].split('.')[0])
    correct_cols = ['image_filename', 'prop_food', 'prop_place', 'prop_customer', 'prop_chef_staff', 'wtdi_food', 'wtdi_place', 'wtdi_customer', 'wtdi_chef_staff', 'price_estimate', 'wtdi_food_reason', 'wtdi_place_reason', 'wtdi_customer_reason', 'wtdi_chef_staff_reason', 'price_component', 'brightness', 'colorfulness', 'color_complexity', 'focus', 'depth_of_field', 'contrast', 'saturation', 'sharpness', 'vibrancy', 'num_cust', 'space_size_2d', 'space_size_3d', 'overall_wtdi']
    df_cols = df.columns.to_list()
    correct = True
    target_rows = 20 if batch_num != 250 else 19
    # 수집해야할 사진 개수가 맞는지 확인
    if len(df) != target_rows:
        correct = False
        return correct

    # 모든 칼럼이 존재하는 지 확인
    if len(set(correct_cols).intersection(set(df_cols))) != len(correct_cols):
        correct = False
        return correct

    # 중복된 이미지 파일 이름이 없는지 확인
    if len(df['image_filename'].unique()) != target_rows:
        correct = False
        return correct

    # 잘못된 이미지 파일이 포함되어 있는지 확인
    batch_filenames = batch_dict[batch_num]
    filenames = df['image_filename'].to_list()
    name_checks = []
    for b_filename in batch_filenames:
        b_id = int(b_filename.split('_')[0])
        b_pid = int(b_filename.split('_')[3].split('.')[0])
        b_pos = b_filename.split('_')[2]
        for filename in filenames:
            found = False
            id = int(filename.split('_')[0])
            pid = int(filename.split('_')[3].split('.')[0])
            pos = filename.split('_')[2]
            if id == b_id and pid == b_pid and pos == b_pos:
                found = True
                break

        name_checks.append(1 if found else 0)

    if sum(name_checks) != target_rows:
        correct = False

    # 비율 칼럼과 WTDI 칼럼에 대한 유효성 확인
    cat_cols = ['food', 'place', 'customer', 'chef_staff']
    for _, r in df.iterrows():
        for cat in cat_cols:
            if r[f'prop_{cat}'] == 0:
                if r[f'wtdi_{cat}'] != 'none' or r[f'wtdi_{cat}_reason'] != 'none':
                    correct = False
            else:
                if r[f'wtdi_{cat}'] == 'none' or r[f'wtdi_{cat}_reason'] == 'none':
                    correct = False
        for cat in cat_cols:
            if type(r[f'prop_{cat}']) != float:
                if r[f'prop_{cat}'] != 'none':
                    correct = False
            else:
                if r[f'prop_{cat}'] < 0 or r[f'prop_{cat}'] > 1:
                    correct = False
        if not math.isclose(r['prop_food'] + r['prop_place'] + r['prop_customer'] + r['prop_chef_staff'], 1):
            correct = False
    
    # 이미지 스펙 칼럼 확인
    image_spec_cols = ['brightness', 'colorfulness', 'color_complexity', 'focus', 'depth_of_field', 'contrast', 'saturation', 'sharpness', 'vibrancy']
    for _, r in df.iterrows():
        for spec in image_spec_cols:
            if r[spec] > 1 or r[spec] < 0:
                correct = False

    # 손님 수 칼럼 확인
    for _, r in df.iterrows():
        if r['num_cust'] > 0:
            if r['wtdi_customer'] == 'none' or r['prop_customer'] == 0:
                correct = False
        else:
            if r['wtdi_customer'] != 'none' or r['prop_customer'] > 0:
                correct = False

    return correct

def get_mean_sd(x):
    id = x['id'].values[0]
    name = x['name'].values[0]
    position = x['position'].values[0]
    image_filename = x.name
    n = len(x)
    final_dict = {'id': id, 'name': name, 'position': position, 'image_filename': image_filename, 'obs': n}

    stat_dict = {}
    target_cols = ['prop_food', 'prop_place', 'prop_customer', 'prop_chef_staff',
                   'wtdi_food', 'wtdi_place', 'wtdi_customer', 'wtdi_chef_staff',
                   'price_estimate', 'brightness', 'colorfulness', 'color_complexity', 'focus', 'depth_of_field', 'contrast', 'saturation', 'sharpness', 'vibrancy', 'num_cust', 'space_size_2d', 'space_size_3d', 'overall_wtdi']
    for col in target_cols:
        # 1이면 하나만 존재. 0이면 반반씩 존재.
        imbalance_ratio = 1
        none_num = len(x[x[col] == 'none'][col])
        valid_num = n - none_num
        if none_num != 0 and n != none_num:
            p = valid_num / n
            imbalance_ratio = abs(2*p - 1)
        col_x = x[x[col] != 'none'][col].astype(float)
        stat_dict[f'{col}_imr'] = imbalance_ratio
        stat_dict[f'{col}_n'] = len(col_x)
        stat_dict[f'{col}_mean'] = col_x.mean() if valid_num > 0 else 'none'
        stat_dict[f'{col}_std'] = col_x.std() if valid_num > 0 else 'none'
    final_dict.update(stat_dict)
    return pd.Series(final_dict)
    
if __name__ == '__main__':
    batch_images = [f for f in os.listdir(RESOURCE_DIR) if f.lower().endswith('jpg')]
    image_info = [{'id': int(f.split('_')[0]), 'pos': f.split('_')[2], 'pid': int(f.split('_')[3].split('.')[0]), 'image_filename': f} for f in batch_images]
    image_info_df = pd.DataFrame(image_info)
    if args.mode == 'stat':
        target_outputs = [pd.read_csv(f) for f in os.listdir('./') if f.startswith('wtdi_eval_by_ChatGPT_5_seed')]
        image_filenames = image_info_df['image_filename'].unique().tolist()
        for output in target_outputs:
            this_image_filenames = output['image_filename'].unique().tolist()
            for inm in this_image_filenames:
                if inm not in image_filenames:
                    name_component = inm.split('_')
                    id = int(name_component[0])
                    pos = name_component[2]
                    pid = int(name_component[3].split('.')[0])
                    base_nm = image_info_df[(image_info_df['id'] == id) & (image_info_df['pos'] == pos) & (image_info_df['pid'] == pid)]['image_filename'].values[0]
                    output.loc[output[(output['id'] == id) & (output['position'] == pos) & (output['pid'] == pid)].index, 'image_filename'] = base_nm
                    print(f'{inm} → {base_nm}') 

        all_df = pd.concat(target_outputs)
        all_df.sort_values(by=['id', 'name', 'position', 'image_filename'], inplace=True)
        stat_df = all_df.groupby('image_filename').progress_apply(get_mean_sd, include_groups=False).reset_index(drop=True)
        stat_df.to_csv(f'wtdi_eval_by_ChatGPT_stats_obs_{len(target_outputs)}.csv', index=False)
            
    else:
        seed = args.seed
        if args.mode == 'eval':
            if seed == 'all':
                print('Unsupported seed option ''all'' for ''eval'' mode.')
                exit()
        
        if args.seed != 'all':
            try:
                seed = int(seed)
            except:
                print('You must enter the integer for the seed number.')
                exit()
        
        if seed == -1:
            if args.mode == 'ec':
                print(f'Seed number must be provided to activate correction mode.')
                exit()
            # Seed 번호 랜덤 생성
            additional_rn = random.randint(1, 100000000000)
            random.seed(time.time_ns() + additional_rn)
            seed = random.randint(10**15, 10**16 - 1)
            print(f'New seed number {seed} has been generated.')
        
        output_dir = f'./output_{seed}'
        if args.mode in ['ec', 'agg'] and args.seed != 'all' and not os.path.isdir(f'./output_{seed}'):
            print(f'The output folder for seed {seed} cannot be found.')
            exit()

        if args.seed != 'all':
            batch_dict = {}
            random.seed(seed)
            random.shuffle(batch_images)

            chunk_size = 20

            for i in range(0, len(batch_images), chunk_size):
                key = i // chunk_size + 1
                batch_dict[key] = batch_images[i:i+chunk_size]

        if args.mode == 'eval':
            # Output 폴더 생성
            if args.overwrite:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)
                print(f'Seed #{seed} Initialized.')
            else:
                if not os.path.exists(output_dir):
                    print('The output folder corresponding the seed number does not exist.')
                    exit()
                print(f'Seed #{seed} Found.')

            failed_list = []
            success_list = []
            df_list = []

            # 진행 중이였던 작업인지 확인.
            output_files = utils.get_output_files(seed)
            if len(output_files) == 0:
                fst_batch = 1
            else:
                print(f'[Seed #{seed}] Found total {len(output_files)} file(s) in the output folder.')
                fst_batch = max([int(f.replace('batch_', '')[:3]) for f in output_files]) + 1

            for batch_num in range(fst_batch, 250 + 1):
                trial_num = 1
                success = False
                print(f'[Seed #{seed}][Batch #{batch_num}] Getting started to analyze... ({trial_num})')
                generated, file = image_eval(seed, batch_num, batch_dict[batch_num])
                while not success or not generated:
                    if generated:
                        this_df = pd.read_csv(f'./output_{seed}/{file}')
                        valid = df_check(this_df, file, batch_dict)
                        if valid:
                            break
                    trial_num += 1
                    print(f'[Seed #{seed}][Batch #{batch_num}] Re-try to evaluate... ({trial_num})')
                    generated, file = image_eval(seed, batch_num, batch_dict[batch_num])

                    
        elif args.mode == 'ec':
            seed_list = []
            if args.seed == 'all':
                seed_list = [int(f.split('_')[1]) for f in os.listdir('./') if f.startswith('output_')]
            else:
                seed_list = [seed]

            for s in seed_list:
                output_files = utils.get_output_files(s)
                df_list = []
                invalid_list = []
                valid_list = []
                omitted_list = []

                if args.seed == 'all':
                    batch_dict = {}
                    batch_images_copy = batch_images.copy()
                    random.seed(s)
                    random.shuffle(batch_images_copy)
                    chunk_size = 20

                    for i in range(0, len(batch_images_copy), chunk_size):
                        key = i // chunk_size + 1
                        batch_dict[key] = batch_images_copy[i:i+chunk_size]

                # 유효성 확인
                for file in output_files:
                    batch_num = int(file.split('_')[1].split('.')[0])
                    match = re.match(r"(batch_\d+)", file)

                    this_df = pd.read_csv(f'./output_{s}/{file}')
                    correct = df_check(this_df, file, batch_dict)
                    if not correct:
                        invalid_list.append(batch_num)
                        continue
                    valid_list.append(batch_num)
                    this_df['batch_num'] = batch_num
                    df_list.append(this_df)

                if len(invalid_list) > 0:
                    print(f'[Seed #{s}] Found total #{len(invalid_list)} invalid batch(s): {', '.join(list(map(str, invalid_list)))}.')
                else:
                    print(f'[Seed #{s}] No file is invalid.')

                # 결측 파일 확인
                for i in range(1, 250 + 1):
                    if i not in valid_list + invalid_list:
                        omitted_list.append(i)
                if len(omitted_list) > 0:
                    print(f'[Seed #{s}] Found total #{len(omitted_list)} missing batch(s): {', '.join(list(map(str, omitted_list)))}.')
                else:
                    print(f'[Seed #{s}] No file is missing.')

                if len(invalid_list + omitted_list) > 0:
                    re_eval_list = sorted(invalid_list + omitted_list)
                    for batch_num in re_eval_list:
                        trial_num = 1
                        corrected = False
                        print(f'[Seed #{s}][Batch #{batch_num}] Re-try to evalueate... ({trial_num})')
                        generated, file = image_eval(s, batch_num, batch_dict[batch_num])
                        while not corrected or not generated:
                            if generated:
                                this_df = pd.read_csv(f'./output_{s}/{file}')
                                valid = df_check(this_df, file, batch_dict)
                                if valid:
                                    break
                            trial_num += 1
                            print(f'[Seed #{s}][Batch #{batch_num}] Re-try to evaluate... ({trial_num})')
                            generated, file = image_eval(s, batch_num, batch_dict[batch_num])
                else:
                    print(f'[Seed #{s}] All files exist and are valid!')

        elif args.mode == 'agg':
            seed_list = []
            if args.seed == 'all':
                seed_list = [int(f.split('_')[1]) for f in os.listdir('./') if f.startswith('output_')]
            else:
                seed_list = [seed]
            for s in seed_list:
                tot_df_list = [(pd.read_csv(f'./output_{s}/{file}'), int(file.split('_')[1].split('.')[0])) for file in utils.get_output_files(s)]
                for this_df, batch_num in tot_df_list:
                    this_df['batch_num'] = batch_num
                if len(tot_df_list) != 250:
                    print(f'[Seed #{s}] Insufficient number of outputs. Expected: 250, Actual: {len(tot_df_list)}.')
                    exit()
                agg_df = pd.concat([df for df, _ in tot_df_list])
                agg_df[['id', 'position', 'pid']] = agg_df["image_filename"].str.extract(
                    r"^(\d+)_.*?_(header|body)_(\d+)\.jpg$"
                )
                if len(agg_df[pd.isna(agg_df['id'])]) > 0:
                    print(agg_df[pd.isna(agg_df['id'])]['batch_num'])
                agg_df["id"] = agg_df["id"].astype(int)
                agg_df["pid"] = agg_df["pid"].astype(int)
                res_name_list = pd.read_csv('focal_res_media_list.csv')
                res_name_list = res_name_list.drop_duplicates(subset=['focal_res_id', 'name'])
                agg_df = agg_df.merge(res_name_list[['focal_res_id', 'name']], left_on='id', right_on='focal_res_id', how='left')
                export_cols = ['id', 'name', 'position', 'pid'] + [col for col in agg_df.columns.to_list()[:-5]]
                agg_df = agg_df[export_cols].sort_values(by=['id', 'position', 'pid'], ascending=[True, False, True]).reset_index(drop=True)
                if s == 3850040609873900:
                    batch_dict = {}
                    batch_images_copy = batch_images.copy()
                    random.seed(s)
                    random.shuffle(batch_images_copy)
                    chunk_size = 20

                    for i in range(0, len(batch_images_copy), chunk_size):
                        key = i // chunk_size + 1
                        batch_dict[key] = batch_images_copy[i:i+chunk_size]
                    img_list = agg_df[agg_df['batch_num'] == 199]['image_filename'].to_list()
                    for f in batch_dict[199]:
                        if f not in img_list:
                            print(f)
                agg_df.drop_duplicates(subset='image_filename', inplace=True)
                if len(agg_df) != len(batch_images):
                    print(f'[Seed #{s}] The length of the aggregated dataframe does not match with the number of the images. Expected: {len(batch_images)}, Actual: {len(agg_df)}.')
                    exit()
                agg_df.to_csv(f'wtdi_eval_by_ChatGPT_5_seed_{s}.csv', encoding='utf-8-sig', index=False)
        
        else:
            print(f'{args.mode} is an unsupported mode. Enther either eval, ec, agg, or stat.')
            exit()
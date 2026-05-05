import os

def upload_file(client, seed, batch_num, image_files):
    file_id_dict = {}
    batch_str = f'[Seed #{seed}][Batch #{batch_num}] '
    file_dict_uploaded = {f.filename: f.id for f in client.files.list()}
    new_image_files = list(set(image_files).difference(set(list(file_dict_uploaded.keys()))))
    if len(new_image_files) == 0:
        print(f'{batch_str}There are no images to be added because they have already been uploaded.')
        file_id_dict = {filename: fid for filename, fid in file_dict_uploaded.items() if filename in image_files}
    else:
        for image in new_image_files:
            file_path = f'./resources/{image}'
            with open(file_path, 'rb') as file_content:
                result = client.files.create(
                    file=file_content,
                    purpose='vision',
                )
                file_id_dict[image] = result.id
        print(f'{batch_str}{len(new_image_files)} image(s) have been uploaded: {", ".join(new_image_files)}')
        dup_files = list(set(list(file_dict_uploaded.keys())).intersection(set(image_files)))
        if len(dup_files) > 0:
            print(f'{batch_str}{len(image_files) - len(new_image_files)} image(s) is/are found to exist: {", ".join(dup_files)}')
            for image in dup_files:
                if image not in file_id_dict:
                    file_id_dict[image] = file_dict_uploaded[image]
    return file_id_dict

def get_unique_filename(base_path, batch_num, prefix='batch_', ext='.csv'):
    counter = 0
    while True:
        suffix = f'{counter:03d}' if counter > 0 else ''
        filename = f"{prefix}{batch_num:03d}{'_' + suffix if suffix else ''}{ext}"
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path):
            return full_path, filename
        counter += 1
        
def get_output_files(seed):
    output_files = []
    for batch_num in range(1, 250 + 1):
        batch_output_files = [f for f in os.listdir(f'./output_{seed}/') if f.startswith(f'batch_{batch_num:03d}') and f.endswith('.csv')]
        if len(batch_output_files) == 0:
            continue
        lastest_filename = batch_output_files[0]
        if len(batch_output_files) > 1:
            batch_output_filepath = [f'./output_{seed}/{file}' for file in batch_output_files]
            lastest_ctime = os.path.getctime(batch_output_filepath[0])
            for i in range(1, len(batch_output_files)):
                this_ctime = os.path.getctime(batch_output_filepath[i])
                if lastest_ctime < this_ctime:
                    lastest_ctime = this_ctime
                    lastest_filename = batch_output_files[i]
        output_files.append(lastest_filename)
    return output_files
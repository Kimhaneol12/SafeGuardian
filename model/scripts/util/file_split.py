import os
import shutil

def move_sensor_files(home_directory, xlsx_directory):
    n_directory = os.path.join(home_directory, 'VIDEO\\N\\N')
    by_directory = os.path.join(home_directory, 'VIDEO\\Y\\BY')
    fy_directory = os.path.join(home_directory, 'VIDEO\\Y\\FY')
    sy_directory = os.path.join(home_directory, 'VIDEO\\Y\\SY')

    search_directory = [n_directory, by_directory, fy_directory, sy_directory]

    for directory in search_directory:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.xlsx'):
                    source_file_path = os.path.join(root, file)
                    target_file_path = os.path.join(xlsx_directory, file)
                    os.makedirs(xlsx_directory, exist_ok=True)
                    shutil.move(source_file_path, target_file_path)

    sensor_files = os.listdir(xlsx_directory)
    
    for file in sensor_files:
        if '_C1' not in file:
            os.remove(os.path.join(xlsx_directory, file))
            print(f"Removed: {file}")

## 사용방법

# home_directory = os.getcwd()
# xlsx_directory = 'D:/NIA/2023-09-06/raw_sensor'
# move_sensor_files(home_directory, xlsx_directory)

# util 폴더에 넣기
# from util.file_extractor import move_sensor_files
# 현재 홈 디렉토리와 센서 파일이 저장될 폴더를 지정하고, move_sensor_files(홈 디렉토리, 타겟 디렉토리)로 하면 됨
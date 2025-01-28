"""
File to preprocess the fonts data. We want to get them to into a train and test
folders that pytorch Imagefolder can work with.
"""

import os
import random
import shutil
import tarfile

import kagglehub

output_dir = os.getcwd() + "/fonts"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
TEST_SIZE = 0.3

# Download latest version
tarfile_path = kagglehub.dataset_download(
    "supreethrao/chars74kdigitalenglishfont"
)

with tarfile.open(os.path.join(tarfile_path, "EnglishFnt.tgz"), "r:gz") as tar:
    members = tar.getmembers()
    for member in members:
        if member.path.startswith("English/Fnt/"):
            member.path = member.path.replace("English/Fnt/", "")
    tar.extractall(path=output_dir, members=members)

folder_mappers = {
    "Sample001": "0",
    "Sample002": "1",
    "Sample003": "2",
    "Sample004": "3",
    "Sample005": "4",
    "Sample006": "5",
    "Sample007": "6",
    "Sample008": "7",
    "Sample009": "8",
    "Sample010": "9",
    "Sample011": "A",
    "Sample012": "B",
    "Sample013": "C",
    "Sample014": "D",
    "Sample015": "E",
    "Sample016": "F",
    "Sample017": "G",
    "Sample018": "H",
    "Sample019": "I",
    "Sample020": "J",
    "Sample021": "K",
    "Sample022": "L",
    "Sample023": "M",
    "Sample024": "N",
    "Sample025": "O",
    "Sample026": "P",
    "Sample027": "Q",
    "Sample028": "R",
    "Sample029": "S",
    "Sample030": "T",
    "Sample031": "U",
    "Sample032": "V",
    "Sample033": "W",
    "Sample034": "X",
    "Sample035": "Y",
    "Sample036": "Z",
    "Sample037": "small_a",
    "Sample038": "small_b",
    "Sample039": "small_c",
    "Sample040": "small_d",
    "Sample041": "small_e",
    "Sample042": "small_f",
    "Sample043": "small_g",
    "Sample044": "small_h",
    "Sample045": "small_i",
    "Sample046": "small_j",
    "Sample047": "small_k",
    "Sample048": "small_l",
    "Sample049": "small_m",
    "Sample050": "small_n",
    "Sample051": "small_o",
    "Sample052": "small_p",
    "Sample053": "small_q",
    "Sample054": "small_r",
    "Sample055": "small_s",
    "Sample056": "small_t",
    "Sample057": "small_u",
    "Sample058": "small_v",
    "Sample059": "small_w",
    "Sample060": "small_x",
    "Sample061": "small_y",
    "Sample062": "small_z",
}
for root, dirs, files in os.walk(output_dir):
    for folder in dirs:
        if folder in folder_mappers:
            old_path = os.path.join(root, folder)
            new_path = os.path.join(root, "train", folder_mappers[folder])
            os.rename(old_path, new_path)

for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    if os.path.isdir(class_path):
        files = os.listdir(class_path)
        num_test = int(len(files) * TEST_SIZE)
        test_files = random.sample(files, num_test)

        # Create test directory for this class
        test_class_path = os.path.join(test_dir, class_folder)
        os.makedirs(test_class_path, exist_ok=True)

        # Move selected files to test directory
        for file in test_files:
            old_path = os.path.join(class_path, file)
            new_path = os.path.join(test_class_path, file)
            os.rename(old_path, new_path)
try:
    shutil.rmtree(os.path.dirname(tarfile_path))
    shutil.rmtree(os.path.join(output_dir, "English"))
except FileNotFoundError:
    print("File/Folders not found")

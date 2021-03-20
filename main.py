import os
from model import estimate_model, predict
import config

if __name__ == "__main__":
    image_list_yes = os.listdir(config.FOLDER_NAME_YES)
    image_list_yes = [config.FOLDER_NAME_YES + i for i in image_list_yes]
    image_list_no = os.listdir(config.FOLDER_NAME_NO)
    image_list_no = [config.FOLDER_NAME_NO + i for i in image_list_no]
    image_list = image_list_yes + image_list_no

    correct_ans = []
    for i in range(len(image_list_yes)):
        correct_ans.append(True)
    for i in range(len(image_list_no)):
        correct_ans.append(False)

    print(f"Accuracy is {estimate_model(image_list, correct_ans, False):0.2f}")
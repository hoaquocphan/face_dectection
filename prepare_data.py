import os
from os import walk
import argparse
import math
from shutil import copyfile
import shutil

os.mkdir("train") 
os.mkdir("validation") 
os.mkdir("test")
os.mkdir("temp")

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=str, default='10')  
args = parser.parse_args()



for i in range(1, int(args.n)+1):
    print(i)
    DIR = "VN-celeb/" + str(i)
    DIR2 = "temp/" + str(i)
    os.mkdir(DIR2)
    number_image=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    print(number_image)
    j=0
    image_name = []
    for (dirpath, dirnames, filenames) in walk(DIR):
        image_name.extend(filenames)
    for image in image_name:
        image_file = DIR + "/" + image
        output_name= DIR2 + "/" + str(i) + "_" + str(j) + ".png"
        copyfile(image_file, output_name)
        j=j+1

for i in range(1, int(args.n)+1):
    print(i)
    DIR = "temp/" + str(i)
    train_dir = "train/" + str(i)
    validation_dir = "validation/" + str(i)
    test_dir = "test/" + str(i)
    number_image=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if number_image == 2:
        input_name= DIR + "/" + str(i) + "_0" + ".png"
        output_name= train_dir + "_0" + ".png"
        os.rename(input_name, output_name)
        input_name= DIR + "/" + str(i) + "_1" + ".png"
        output_name= validation_dir + "_1" + ".png"
        os.rename(input_name, output_name)
    else:
        input_name= DIR + "/" + str(i) + "_" + str(number_image-1) + ".png"
        output_name= test_dir + "_" + str(number_image-1) + ".png"
        os.rename(input_name, output_name)
        num_valid = math.ceil((number_image-1)/9)
        num_train = number_image-1 - num_valid
        for j in range(0, num_train):
            input_name= DIR + "/" + str(i) + "_" + str(j) + ".png"
            output_name= train_dir + "_" + str(j) + ".png"
            os.rename(input_name, output_name)
        for j in range(num_train, num_train + num_valid):
            input_name= DIR + "/" + str(i) + "_" + str(j) + ".png"
            output_name= validation_dir + "_" + str(j) + ".png"
            os.rename(input_name, output_name)

shutil.rmtree("temp",ignore_errors=True)
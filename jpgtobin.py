import cv2
import numpy as np
import os, sys
import pickle
from tqdm import tqdm

path = "./Data/dataset/test/"

file_list = os.listdir(path)

print(len(file_list))

set_of_patch = []

for fn in tqdm(file_list):
    img = cv2.imread(path + fn, cv2.IMREAD_COLOR)
    #img = np.array(img)
    set_of_patch.append(img)

set_of_patch = np.array(set_of_patch)
print(set_of_patch.shape)
print("making pickle file")
fo = open(path + "test.pkl", 'wb')
pickle.dump(set_of_patch, fo, pickle.HIGHEST_PROTOCOL)
fo.close()
print("pikle file is made")

fo = open(path + "test.pkl" , 'rb')
dataset = pickle.load(fo)
fo.close()
print(dataset.shape)


print("program end")


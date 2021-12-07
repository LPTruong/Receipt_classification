import pickle
import pandas as pd
import cv2

from variable import _dir, pretrain_dir

df = pd.read_csv(pretrain_dir + "label_highland.csv")

# Final ML model with full datas
ridge_path = pretrain_dir + "ridge_highland.sav"

loaded_model = pickle.load(open(ridge_path, 'rb'))

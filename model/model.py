import pickle
import pandas as pd
import cv2

from variable import _dir, highland_dir

df = pd.read_csv(highland_dir + "label_highland.csv")

# Final ML model with full datas
ridge_path = highland_dir + "ridge_highland.sav"

loaded_model = pickle.load(open(ridge_path, 'rb'))

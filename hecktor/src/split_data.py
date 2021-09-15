import pickle
import random
import pandas as pd
import json


bb_csv_path = r"/home/server1080/2.0T/lt/HECKTOR/hecktor2021_train/hecktor2021_bbox_training.csv"
bb = pd.read_csv(bb_csv_path)
patientID_list = list(bb.PatientID)

train_patient = int(len(patientID_list)*0.8)
val_patient = len(patientID_list) - train_patient
train_patient_list = random.sample(patientID_list, train_patient)
val_patient_list = random.sample(patientID_list, val_patient)

dicts = {}
for p in train_patient_list:
    dicts.setdefault('train', []).append(p)
for p in val_patient_list:
    dicts.setdefault('val', []).append(p)

save_split_path = r"/home/server1080/Documents/lt/HECKTOR/hecktor/src/data/splits/split_9.pkl"
with open(save_split_path, 'w', encoding='utf-8') as f:
    json.dump(dicts, f)

with open(save_split_path) as f:
    out = json.load(f)

f.close()
print(out)

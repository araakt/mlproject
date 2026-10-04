import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        model_names = list(models.keys())
        for i, model_name in enumerate(model_names, start=1):
            model = models[model_name]
            para = param[model_name]

            print(f"[{i}/{len(model_names)}] Searching parameters for {model_name}...", flush=True)
            gs = GridSearchCV(model, para, cv=3, verbose=1)
            gs.fit(X_train,y_train)

            print(f"[{i}/{len(model_names)}] Best parameters for {model_name}: {gs.best_params_}", flush=True)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            print(f"[{i}/{len(model_names)}] Finished {model_name}; test R2={test_model_score:.4f}", flush=True)

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
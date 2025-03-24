import os
import gc
from memory_profiler import profile


import numpy as np
import pandas as pd
import tqdm 

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

from utils import filter_features_by_missing_rate, impute_missing_values 

def create_pipeline():
    print("Creating pipeline......")
    class_weight = {0: 1, 1: 5}
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            class_weight=class_weight, 
            random_state=42,
            max_features='sqrt',
            n_jobs=-1  
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=4,
            subsample=0.8
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, 
            scale_pos_weight=len(y[y==0])/max(1, len(y[y==1])) * 2,  
            random_state=42,
            max_depth=4,
            learning_rate=0.1,
            n_jobs=-1  
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100, 
            class_weight=class_weight, 
            random_state=42,
            max_depth=4,
            n_jobs=-1 
        )
    }

    pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipelines[name] = pipeline

    return pipelines
def train(X, y):
    
    print("Training......")
    pipelines = create_pipeline()
    # for name, pipeline in pipelines.items():
        

def main():
    path = "D:\\aigo\stock_rocket_prediction\dataset\\train\\training.csv"
    summart_dict = dict()
    label_name = '飆股'
    # Loading dataset
    
    def process_data():
        nonlocal summart_dict
        stock_data, features_list = prepare_data(path)

        summart_dict["The shape of DataFrame"] = stock_data.shape
        summart_dict["The number of features"] = len(features_list)
        return filter_features_by_missing_rate(stock_data, threshold= 0, return_type='df')
    
    filter_stock_data = process_data()
    print(f"The shape of filter DataFrame: {filter_stock_data.shape}")
    
    # Impute missing value
    filter_stock_data = impute_missing_values(filter_stock_data)

    # 
    X = filter_stock_data.drop([label_name,'ID'], axis=1)
    y = filter_stock_data[label_name]
    print(f"最終數據集: {X.shape[0]} 樣本, {X.shape[1]} 特徵")
    print(f"標籤分布: 正例 {y.sum()} ({y.sum()/len(y)*100:.2f}%), 負例 {len(y)-y.sum()} ({(1-y.sum()/len(y))*100:.2f}%)")
    del filter_stock_data

    # train(X,y)
    

if __name__ == "__main__":
    main()
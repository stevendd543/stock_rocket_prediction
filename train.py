import os
import gc
from memory_profiler import profile


import numpy as np
import pandas as pd
import tqdm 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import clone

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

from utils import *
def create_pipeline(y):
    print("Creating pipeline......")
    class_weight = {0: 1, 1: 10}
    models = {
        # 'RandomForest': RandomForestClassifier(
        #     n_estimators=100, 
        #     class_weight=class_weight, 
        #     random_state=42,
        #     max_features='sqrt',
        #     n_jobs=-1  
        # ),
        # 'GradientBoosting': GradientBoostingClassifier(
        #     n_estimators=100, 
        #     random_state=42,
        #     max_depth=4,
        #     subsample=0.8
        # ),
        # 'XGBoost': xgb.XGBClassifier(
        #     n_estimators=100, 
        #     # scale_pos_weight=len(y[y==0])/max(1, len(y[y==1])) * 2,  
        #     scale_pos_weight=120,  

        #     random_state=42,
        #     max_depth=4,
        #     # learning_rate=0.1,
        #     n_jobs=-1  
        # ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=500,               
            class_weight=class_weight,        
            
            # scale_pos_weight=133,         # 負/正
            random_state=42,
            max_depth=10,                     
            learning_rate=0.09,              
            num_leaves=64,                   
            min_child_samples=20,            
            subsample=0.2,                   
            colsample_bytree=0.8,            
            reg_alpha=0.1,                   
            reg_lambda=1.0,                  
            n_jobs=-1,                      
            # is_unbalance=True,               
            boost_from_average=False,         
            verbose=-1,
            
        )
    }
    result = {}
    pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('model', model)
        ])
        pipelines[name] = pipeline

    return pipelines
    
from sklearn.model_selection import cross_val_score

def evaluate_feature_importance(X, y, pipelines=None, test_size=0.2, smote=False, thr=None):
    
    if pipelines is None:
        pipelines = train_and_evaluate(X,y, smote=True, thr = thr)
    
    
    model_name = list(pipelines.keys())[0]
    baseline_model = pipelines[model_name]
    
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X, y, test_size=test_size, random_state=42)
    if smote:
        X_train_base, y_train_base = apply_oversampling(X_train_base, y_train_base, method='SMOTE')
    
    y_pred_base = baseline_model.predict(X_test_base)
    y_prob_base = baseline_model.predict_proba(X_test_base)[:, 1]

    cr = classification_report(y_test_base, y_pred_base, output_dict=True)

    baseline_metrics = {
        'f1': cr(y_test_base, y_pred_base, output_dict=True)['1']['f1-score']  if '1' in cr.keys() else cr['1.0']['f1-score'],
        'roc_auc': roc_auc_score(y_test_base, y_prob_base)
    }
    
    print(f"Performace of baseline:")
    print(f"  F1: {baseline_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
    
    # 逐一移除特徵評估
    importance_results = []
    for feature_idx, feature in enumerate(X.columns):
        X_reduced = X.drop(columns=[feature])
        print(f"\nEvaluate feature {feature} ({feature_idx+1}/{len(X.columns)}):")
        
        X_train, X_test, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=test_size, random_state=42)
        if smote:
            X_train, y_train_r = apply_oversampling(X_train, y_train_r, method='SMOTE')
        
        temp_model = clone(baseline_model)
        temp_model.fit(X_train, y_train_r)
        
        y_pred = temp_model.predict(X_test)
        y_prob = temp_model.predict_proba(X_test)[:, 1]
        
        cr = classification_report(y_test_r, y_pred, output_dict=True)
        cm = confusion_matrix(y_test_r, y_pred)
   
        current_f1 = cr['1']['f1-score'] if '1' in cr.keys() else cr['1.0']['f1-score']
        current_roc = roc_auc_score(y_test_r, y_prob)
        
        importance_f1 = baseline_metrics['f1'] - current_f1
        importance_roc = baseline_metrics['roc_auc'] - current_roc
        
        print(f"  Reduced version F1: {current_f1:.4f} (Variation: {importance_f1:.4f})")
        print(f"  Reduced version ROC-AUC: {current_roc:.4f} (Variation: {importance_roc:.4f})")
        
        importance_results.append({
            'model_name': model_name,
            'removed_feature': feature,
            'f1_score': current_f1,
            'f1_change': importance_f1,
            'roc_auc': current_roc,
            'roc_auc_change': importance_roc,
            'missing_value_threshold': thr,
            'feature_count': X_reduced.shape[1],
            'confusion_matrix_tn': int(cm[0,0]),
            'confusion_matrix_fp': int(cm[0,1]),
            'confusion_matrix_fn': int(cm[1,0]),
            'confusion_matrix_tp': int(cm[1,1])
        })
    
    results_df = pd.DataFrame(importance_results)
    feature_importance_path = 'feature_importance_results.csv'
    
    if os.path.exists(feature_importance_path):
        results_df.to_csv(feature_importance_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(feature_importance_path, index=False)
    
    print(f"\n特徵重要性評估結果已儲存至 {feature_importance_path}")
    
    
    top_features = results_df.sort_values('f1_change', ascending=False)
    print("\n特徵重要性排名 (根據F1分數變化):")
    for idx, row in top_features.head(10).iterrows():
        print(f"{idx+1}. {row['removed_feature']} - F1變化: {row['f1_change']:.4f}")
    
    return results_df

def train_and_evaluate(X, y, test_size=0.15, smote = False, thr = 0, feature_importance_path = None):
    print("Training and Evaluating......")
    if feature_importance_path is not None:
        print("Dropping by f1.....")
        feature_importance_df = pd.read_csv(feature_importance_path)
        feature_to_keep = drop_features_by_f1_change(feature_importance_df, 0.1)
        X = X[feature_to_keep]
        print(f"X length = {len(feature_to_keep)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    pipelines = create_pipeline(y_train)
    
    if smote:
        print(f"Applying smote with shape {X.shape}...")
        X_train, y_train = apply_oversampling(X_train, y_train, method='SMOTE')
    # pipelines['XGBoost'].set_params(model__scale_pos_weight=len(y_train[y_train==0])/max(1, len(y_train[y_train==1])) * 2)
    results = []
    
    for name, pipeline in pipelines.items():
        print(f"\nCross-Validation for {name}:")
        # scores = cross_validate(pipeline, X_train, y_train, cv=5, scoring=['f1', 'precision', 'recall'])
        # print(f"Mean F1: {scores['test_f1'].mean():.4f} (+/- {scores['test_f1'].std() * 2:.4f})")
        # print(f"Mean Precision: {scores['test_precision'].mean():.4f}")
        # print(f"Mean Recall: {scores['test_recall'].mean():.4f}")
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        model_path = os.path.join('models',f'model_{name}')
        model_data = {
            'model': pipeline,
            'feature_names': X_test.columns.tolist()
        }
        joblib.dump(model_data, model_path)

        print(f"\nTest定義模型在測試集上的表現 {name}:")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            'TN': int(cm[0, 0]),
            'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]),
            'TP': int(cm[1, 1])
        }
        
        cr = classification_report(y_test, y_pred, output_dict=True)
        print(cr)
        result = {
            'model_name': name,
            'f1_score': cr['1']['f1-score'] if '1' in cr.keys() else cr['1.0']['f1-score'],
            'precision': cr['1']['precision'] if '1' in cr.keys() else cr['1.0']['precision'],
            'recall': cr['1']['recall'] if '1' in cr.keys() else cr['1.0']['recall'],
            'missing value threshold': thr,
            'feature_count': X.shape[1],
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix_tn': cm_dict['TN'],
            'confusion_matrix_fp': cm_dict['FP'],
            'confusion_matrix_fn': cm_dict['FN'],
            'confusion_matrix_tp': cm_dict['TP']
        }
        results.append(result)
    
        results_df = pd.DataFrame(results)
        eval_csv_path = 'model_evaluation_results.csv'
        
        if os.path.exists(eval_csv_path):
            results_df.to_csv(eval_csv_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(eval_csv_path, index=False)
        
        print(f"\nSaved Evaluation Results to {eval_csv_path}")
    return pipelines

def main():
    path = "example_filled.csv"
    missing_report = "missing_values_report_total.csv"
    missing_report = None
    summart_dict = dict()
    label_name = '飆股'
    missing_thr = 0
    evaluate_importance = False 
    fip = "feature_importance_results.csv" # feature_importance_path
    # fip = None
    # Loading dataset
    
    def process_data():
        nonlocal summart_dict
        nonlocal missing_report
        
        stock_data, features_list = prepare_dataframe(path)
        print(f"The shape of DataFrame : {stock_data.shape}")
        print("Start Processing")
        summart_dict["The shape of DataFrame"] = stock_data.shape
        summart_dict["The number of features"] = len(features_list)
        print("End Processing")
        # return filter_features_by_missing_rate(stock_data, threshold= 0, return_type='df')
        return filter_features_by_missing_rate(df = stock_data, missing_report=missing_report, threshold= missing_thr, return_type='df', )
    
    
    filter_stock_data = process_data()
  
    y = filter_stock_data[label_name]

    filter_stock_data = filter_stock_data.iloc[:, :]
    print(f"The shape of filter DataFrame: {filter_stock_data.shape}")
    
    # Impute missing value
    filter_stock_data = impute_missing_values(filter_stock_data,'median')
    if label_name in filter_stock_data.columns:
        filter_stock_data = filter_stock_data.drop([label_name], axis=1)
    
    
    if 'ID' in filter_stock_data.columns:
        filter_stock_data = filter_stock_data.drop(['ID'], axis=1)

    X = filter_stock_data
    # 

    
    print(f"最終數據集: {X.shape[0]} 樣本, {X.shape[1]} 特徵")
    print(f"標籤分布: 正例 {y.sum()} ({y.sum()/len(y)*100:.2f}%), 負例 {len(y)-y.sum()} ({(1-y.sum()/len(y))*100:.2f}%)")
    del filter_stock_data

    print("Training....")
    trained_pipelines = train_and_evaluate(X,y, smote=True, thr = missing_thr, feature_importance_path= fip)
    
    
    if evaluate_importance:
        print("\nStart to evaluate importance of features...")
        evaluate_feature_importance(X, y, pipelines=trained_pipelines, smote=True, thr=missing_thr)

if __name__ == "__main__":
    main()
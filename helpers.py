import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import CountEncoder
from category_encoders.target_encoder import TargetEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
seed = 42

def get_feature_importance(lst: list):
    for i in range(3):
        feature_names = lst[i].named_steps['preprocessor'].get_feature_names_out()
        if i == 0:
            importance = lst[i].named_steps['model'].coef_
            feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values(by='importance', key=abs, ascending=False)
        else:
            importance = lst[i].named_steps['model'].feature_importances_
            feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(24, 18))
        sns.barplot(data = feature_importance, x = 'importance', y = 'feature', orient='h')
        ax.set_title(lst[i].named_steps['model'].__class__.__name__, fontsize=18)
        ax.set_xlabel('Values', fontsize=14)
        ax.set_ylabel("Features", fontsize=14)
        ax.tick_params(axis='x', rotation=0, labelsize=12)

def feature_importance_xgb(lst: list):
        feature_names = lst[0].named_steps['preprocessor'].get_feature_names_out()
        importance = lst[0].named_steps['model'].feature_importances_
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(24, 18))
        sns.barplot(data = feature_importance, x = 'importance', y = 'feature', orient='h')
        ax.set_title(lst[0].named_steps['model'].__class__.__name__, fontsize=18)
        ax.set_xlabel('Values', fontsize=14)
        ax.set_ylabel("Features", fontsize=14)
        ax.tick_params(axis='x', rotation=0, labelsize=12)


def bin_pipe(df):
    # ====== 1. Разделение на X и y ======
    X = df.drop('price', axis=1)
    y = df['price']
    
    # ====== 2. Определение типов признаков ======
    num_cols = X.select_dtypes(include='number').columns.to_list()
    cat_cols = X.select_dtypes(exclude='number').columns.to_list()
    
    # ====== 3. Препроцессинг ======
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])  
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # ====== 4. Разбиение ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed
    )
    
    # ====== 5. Модели ======
    models = {
        'XGBoost': XGBRegressor(random_state=seed)
    }
    
    # ====== 6. Кросс-валидация ======
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error']

    cv = KFold(n_splits=5, shuffle=True, random_state=seed) 
    
    cross_val_res = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cross_val={}
        for metric in scoring:
            mean_score = abs(cv_results[f'test_{metric}'].mean())
            std_score = abs(cv_results[f'test_{metric}'].std())
            cross_val[metric] = str(round(mean_score, 4)) + ' ± ' + str(round(std_score, 4))
        cross_val_res[name] = cross_val
    
    result_cross_val = pd.DataFrame(cross_val_res).T
    result_cross_val.rename(columns={'neg_mean_squared_error': 'mse', 
                                     'neg_mean_absolute_error': 'mae', 
                                     'neg_mean_absolute_percentage_error':'mape', 
                                     'neg_root_mean_squared_error':'rsme'}, inplace=True)
    
    #====== 7. Финальное обучение и тест ======
    list_of_models=[]
    results={}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        list_of_models.append(pipeline)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse_score = mse(y_test, y_pred)
        mae_score = mae(y_test, y_pred)
        mape_score = mape(y_test, y_pred)
        rmse_score = np.sqrt(mse(y_test, y_pred))
        results[name] = [r2, mse_score, mae_score, mape_score, rmse_score]
    
    result_test = pd.DataFrame(results, index=['r2', 'mse', 'mae', 'mape', 'rsme']).T

    return result_cross_val, result_test, list_of_models

def new_col_pipe(df):
    # ====== 1. Разделение на X и y ======
    X = df.drop('price', axis=1)
    y = df['price']
    
    # ====== 2. Определение типов признаков ======
    num_cols = X.select_dtypes(include='number').columns.to_list()
    cat_cols = X.select_dtypes(exclude='number').columns.to_list()
    
    # ====== 3. Препроцессинг ======
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])  
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # ====== 4. Разбиение ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed
    )
    
    # ====== 5. Модели ======
    models = {
        'XGBoost': XGBRegressor(random_state=seed)
    }
    
    # ====== 6. Кросс-валидация ======
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error']

    cv = KFold(n_splits=5, shuffle=True, random_state=seed) 
    
    cross_val_res = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cross_val={}
        for metric in scoring:
            mean_score = abs(cv_results[f'test_{metric}'].mean())
            std_score = abs(cv_results[f'test_{metric}'].std())
            cross_val[metric] = str(round(mean_score, 4)) + ' ± ' + str(round(std_score, 4))
        cross_val_res[name] = cross_val
    
    result_cross_val = pd.DataFrame(cross_val_res).T
    result_cross_val.rename(columns={'neg_mean_squared_error': 'mse', 
                                     'neg_mean_absolute_error': 'mae', 
                                     'neg_mean_absolute_percentage_error':'mape', 
                                     'neg_root_mean_squared_error':'rsme'}, inplace=True)
    
    #====== 7. Финальное обучение и тест ======
    list_of_models=[]
    results={}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        list_of_models.append(pipeline)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse_score = mse(y_test, y_pred)
        mae_score = mae(y_test, y_pred)
        mape_score = mape(y_test, y_pred)
        rmse_score = np.sqrt(mse(y_test, y_pred))
        results[name] = [r2, mse_score, mae_score, mape_score, rmse_score]
    
    result_test = pd.DataFrame(results, index=['r2', 'mse', 'mae', 'mape', 'rsme']).T

    return result_cross_val, result_test, list_of_models

def target_enc_pipe(df):
    # ====== 1. Разделение на X и y ======
    X = df.drop('price', axis=1)
    y = df['price']
    
    # ====== 2. Определение типов признаков ======
    num_cols = X.select_dtypes(include='number').columns.to_list()
    cat_cols = X.select_dtypes(exclude='number').columns.to_list()
    te_cals = ['cut', 'color', 'clarity']
    
    # ====== 3. Препроцессинг ======
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    te_pipeline = Pipeline([
        ('target_encoder', TargetEncoder()),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
        ('te', te_pipeline, te_cals)
    ])
    
    # ====== 4. Разбиение ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed
    )
    
    # ====== 5. Модели ======
    models = {
        'XGBoost': XGBRegressor(
            random_state=seed
        )
    }
    
    # ====== 6. Кросс-валидация ======
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error']

    cv = KFold(n_splits=5, shuffle=True, random_state=seed) 
    
    cross_val_res = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cross_val={}
        for metric in scoring:
            mean_score = abs(cv_results[f'test_{metric}'].mean())
            std_score = abs(cv_results[f'test_{metric}'].std())
            cross_val[metric] = str(round(mean_score, 4)) + ' ± ' + str(round(std_score, 4))
        cross_val_res[name] = cross_val
    
    result_cross_val = pd.DataFrame(cross_val_res).T
    result_cross_val.rename(columns={'neg_mean_squared_error': 'mse', 
                                     'neg_mean_absolute_error': 'mae', 
                                     'neg_mean_absolute_percentage_error':'mape', 
                                     'neg_root_mean_squared_error':'rsme'}, inplace=True)
    
    #====== 7. Финальное обучение и тест ======
    list_of_models=[]
    results={}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        list_of_models.append(pipeline)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse_score = mse(y_test, y_pred)
        mae_score = mae(y_test, y_pred)
        mape_score = mape(y_test, y_pred)
        rmse_score = np.sqrt(mse(y_test, y_pred))
        results[name] = [r2, mse_score, mae_score, mape_score, rmse_score]
    
    result_test = pd.DataFrame(results, index=['r2', 'mse', 'mae', 'mape', 'rsme']).T

    return result_cross_val, result_test, list_of_models

def outliers_pipe(df):
    
    class QuantileClipper(BaseEstimator, TransformerMixin):
        def __init__(self, lower=0.01, upper=0.99):
            self.lower = lower
            self.upper = upper
            
        def fit(self, X, y=None):
            self.lower_bounds_ = np.quantile(X, self.lower, axis=0)
            self.upper_bounds_ = np.quantile(X, self.upper, axis=0)
            return self
        
        def transform(self, X):
            return np.clip(X, self.lower_bounds_, self.upper_bounds_)

        def get_feature_names_out(self, input_features=None):
            return input_features
    
    # ====== 1. Разделение на X и y ======
    X = df.drop('price', axis=1)
    y = df['price']
    
    # ====== 2. Определение типов признаков ======
    num_cols = X.select_dtypes(include='number').columns.to_list()
    cat_cols = X.select_dtypes(exclude='number').columns.to_list()
    
    # ====== 3. Препроцессинг ======
    num_pipeline = Pipeline([
        ('clipper', QuantileClipper(lower=0.01, upper=0.99)),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])  
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # ====== 4. Разбиение с stratify ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed
    )
    
    # ====== 5. Модели ======
    models = {
        'XGBoost': XGBRegressor(random_state=seed)
    }
    
    # ====== 6. Кросс-валидация ======
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error']

    cv = KFold(n_splits=5, shuffle=True, random_state=seed) 
    
    cross_val_res = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cross_val={}
        for metric in scoring:
            mean_score = abs(cv_results[f'test_{metric}'].mean())
            std_score = abs(cv_results[f'test_{metric}'].std())
            cross_val[metric] = str(round(mean_score, 4)) + ' ± ' + str(round(std_score, 4))
        cross_val_res[name] = cross_val
    
    result_cross_val = pd.DataFrame(cross_val_res).T
    result_cross_val.rename(columns={'neg_mean_squared_error': 'mse', 
                                     'neg_mean_absolute_error': 'mae', 
                                     'neg_mean_absolute_percentage_error':'mape', 
                                     'neg_root_mean_squared_error':'rsme'}, inplace=True)
    
    #====== 7. Финальное обучение и тест ======
    list_of_models=[]
    results={}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        list_of_models.append(pipeline)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse_score = mse(y_test, y_pred)
        mae_score = mae(y_test, y_pred)
        mape_score = mape(y_test, y_pred)
        rmse_score = np.sqrt(mse(y_test, y_pred))
        results[name] = [r2, mse_score, mae_score, mape_score, rmse_score]
    
    result_test = pd.DataFrame(results, index=['r2', 'mse', 'mae', 'mape', 'rsme']).T

    return result_cross_val, result_test, list_of_models




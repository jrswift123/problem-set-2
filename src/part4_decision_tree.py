'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Save dataframe(s) save as .csv('s) in `data/`
'''
# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.preprocessing import LabelEncoder

# Your code here
def decision_tree():
    # read in dfs
    df_arrests_train = pd.read_csv('data/df_arrests_train.csv')
    df_arrests_test = pd.read_csv('data/df_arrests_test.csv')
    
    outcome_column = 'current_charge_felony'
    
    # feature columns
    exclude_cols = [outcome_column, 'pred_lr']
    features = [col for col in df_arrests_train.columns if col not in exclude_cols]
    
    # handle categorical columns
    categorical_cols = df_arrests_train[features].select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df_arrests_train[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # create encoded dataframes
    df_arrests_train_encoded = df_arrests_train.copy()
    df_arrests_test_encoded = df_arrests_test.copy()
    
    # apply encoding to categorical columns
    if categorical_cols:
        train_dummies = pd.get_dummies(df_arrests_train[categorical_cols], prefix=categorical_cols, drop_first=True)
        test_dummies = pd.get_dummies(df_arrests_test[categorical_cols], prefix=categorical_cols, drop_first=True)
        
        df_arrests_train_encoded = df_arrests_train_encoded.drop(columns=categorical_cols)
        df_arrests_test_encoded = df_arrests_test_encoded.drop(columns=categorical_cols)
        
        df_arrests_train_encoded = pd.concat([df_arrests_train_encoded, train_dummies], axis=1)
        df_arrests_test_encoded = pd.concat([df_arrests_test_encoded, test_dummies], axis=1)
        
        df_arrests_test_encoded = df_arrests_test_encoded.reindex(columns=df_arrests_train_encoded.columns, fill_value=0)
    
    # updated features list
    features_encoded = [col for col in df_arrests_train_encoded.columns if col not in exclude_cols]
    
    # prep encoded features
    X_train = df_arrests_train_encoded[features_encoded]
    y_train = df_arrests_train_encoded[outcome_column]
    X_test = df_arrests_test_encoded[features_encoded]
    
    # check for non-numeric data
    non_numeric_cols = []
    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            non_numeric_cols.append(col)
    
    # convert to numeric
    if non_numeric_cols:
        for col in non_numeric_cols:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # fill Nan
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # create parameter grid
    param_grid_dt = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # initialize decision tree
    dt_model = DTC(random_state=42)
    
    # initiliaze grid
    gs_cv_dt = GridSearchCV(
        dt_model, 
        param_grid_dt, 
        cv=5, 
        scoring='accuracy',
        verbose=1,
        n_jobs=-1  # Use all available cores
    )
    
    # run model
    gs_cv_dt.fit(X_train, y_train)
    
    # what was the optimal value for max_depth?
    optimal_depth = gs_cv_dt.best_params_['max_depth']
    print(f"\nOptimal value for max_depth: {optimal_depth}")
    print(f"Best parameters: {gs_cv_dt.best_params_}")
    
    # did it have the most or least regularization? Or in the middle?
    if optimal_depth == min(param_grid_dt['max_depth']):
        print("This max_depth value provides the MOST regularization (smaller depth = simpler tree = more regularization)")
    elif optimal_depth == max(param_grid_dt['max_depth']):
        print("This max_depth value provides the LEAST regularization (larger depth = more complex tree = less regularization)")
    else:
        print("This max_depth value is in the MIDDLE in terms of regularization strength")
    
    # predict for the test set
    df_arrests_test['pred_dt'] = gs_cv_dt.predict(X_test)
    df_arrests_test['pred_dt_proba'] = gs_cv_dt.predict_proba(X_test)[:, 1]
    
    # save df's as csv's
    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

    


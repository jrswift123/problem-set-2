'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: df_arrests, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Your code here
def log_reg():
    # read in df_arrests
    df_arrests = pd.read_csv('data/df_arrests.csv')  # Adjust path as needed

    outcome_column = 'current_charge_felony'  # REPLACE WITH YOUR ACTUAL BINARY OUTCOME COLUMN NAME

    # split df
    df_arrests_train, df_arrests_test = train_test_split(
        df_arrests, 
        test_size=0.3, 
        shuffle=True, 
        stratify=df_arrests[outcome_column],
        random_state=42
    )

    # create features list
    features = ['num_fel_arrests_last_year', 'current_charge_felony'] 

    # create grid
    param_grid = {
        'C': [0.01, 1, 100]  # Three values: small (more regularization), medium, large (less regularization)
    }

    # initialize logistic regression model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)

    # initialize gridsearch
    gs_cv = GridSearchCV(
        lr_model, 
        param_grid, 
        cv=5, 
        scoring='accuracy',  # You can change scoring metric if needed
        verbose=1
    )

    # prep for training
    X_train = df_arrests_train[features]
    y_train = df_arrests_train[outcome_column]

    # run model
    gs_cv.fit(X_train, y_train)

    # find optimal value for c
    optimal_C = gs_cv.best_params_['C']
    print(f"Optimal value for C: {optimal_C}")

    # regularization results
    if optimal_C == min(param_grid['C']):
        print("This C value provides the MOST regularization (smaller C = stronger regularization)")
    elif optimal_C == max(param_grid['C']):
        print("This C value provides the LEAST regularization (larger C = weaker regularization)")
    else:
        print("This C value is in the MIDDLE in terms of regularization strength")

    # predict for the test set
    X_test = df_arrests_test[features]
    df_arrests_test['pred_lr'] = gs_cv.predict(X_test)
    df_arrests_test['pred_dt_proba'] = gs_cv.predict_proba(X_test)[:, 1]


    # save as CSVs
    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

    print(f"Training set shape: {df_arrests_train.shape}")
    print(f"Test set shape: {df_arrests_test.shape}")
    print(f"Best cross-validation score: {gs_cv.best_score_:.4f}")
    print(f"Test set predictions added to df_arrests_test['pred_lr']")


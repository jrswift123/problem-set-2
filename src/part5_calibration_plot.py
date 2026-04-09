'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):

    # calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # create plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()


def calibration_analysis():
    # read in data
    df_arrests_test = pd.read_csv('data/df_arrests_test.csv')
    
    outcome_column = 'current_charge_felony'
    
    # fetch labels
    y_true = df_arrests_test[outcome_column]
    
    
    # create calibration curve for lr
    if 'pred_lr_proba' in df_arrests_test.columns:
        y_prob_lr = df_arrests_test['pred_lr_proba']
        calibration_plot(y_true, y_prob_lr, n_bins=5)
    else:
        prob_cols = [col for col in df_arrests_test.columns if 'proba' in col.lower() or 'prob' in col.lower()]
        if prob_cols:
            print(f"Found probability columns: {prob_cols}")
        else:
            y_prob_lr = df_arrests_test['pred_lr'] if 'pred_lr' in df_arrests_test.columns else None
            if y_prob_lr is not None:
                y_prob_lr = y_prob_lr * 0.8 + 0.1
                calibration_plot(y_true, y_prob_lr, n_bins=5)
    
    # create calibration curve for decision tree
    if 'pred_dt_proba' in df_arrests_test.columns:
        y_prob_dt = df_arrests_test['pred_dt_proba']
        calibration_plot(y_true, y_prob_dt, n_bins=5)
    else:
        if 'pred_dt' in df_arrests_test.columns:
            y_prob_dt = df_arrests_test['pred_dt']
            y_prob_dt = y_prob_dt * 0.8 + 0.1  # Not accurate, just for demo
            calibration_plot(y_true, y_prob_dt, n_bins=5)
    
    # which model is more calibrated?
    # if both probability columns exist compare them
    if 'pred_lr_proba' in df_arrests_test.columns and 'pred_dt_proba' in df_arrests_test.columns:
        lr_bin_means, lr_prob_true = calibration_curve(y_true, df_arrests_test['pred_lr_proba'], n_bins=5)
        dt_bin_means, dt_prob_true = calibration_curve(y_true, df_arrests_test['pred_dt_proba'], n_bins=5)
        
        lr_error = np.mean((lr_prob_true - lr_bin_means) ** 2)
        dt_error = np.mean((dt_prob_true - dt_bin_means) ** 2)
        
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)
        plt.plot(lr_prob_true, lr_bin_means, marker='o', label="Logistic Regression", linewidth=2, markersize=8)
        plt.plot(dt_prob_true, dt_bin_means, marker='s', label="Decision Tree", linewidth=2, markersize=8)
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Comparison: Logistic Regression vs Decision Tree")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        if lr_error < dt_error:
            print("\nAnswer: The Logistic Regression model is MORE calibrated (has lower calibration error).")
            print("Reason: Logistic regression is inherently designed to produce well-calibrated probabilities,")
            print("while decision trees tend to produce probabilities that are more extreme and less calibrated.")
        elif dt_error < lr_error:
            print("\nAnswer: The Decision Tree model is MORE calibrated (has lower calibration error).")
        else:
            print("\nAnswer: Both models have similar calibration.")
            
    elif 'pred_lr_proba' in df_arrests_test.columns:
        print("\nAnswer: Only Logistic Regression probability data available. Cannot compare.")
    elif 'pred_dt_proba' in df_arrests_test.columns:
        print("\nAnswer: Only Decision Tree probability data available. Cannot compare.")
    else:
        print("\nAnswer: No probability data available for comparison.")
        print("\nTo fix this, please update PART 3 and PART 4 to save probability columns:")
        print("In PART 3, add: df_arrests_test['pred_lr_proba'] = gs_cv.predict_proba(X_test)[:, 1]")
        print("In PART 4, add: df_arrests_test['pred_dt_proba'] = gs_cv_dt.predict_proba(X_test)[:, 1]")
    
    return df_arrests_test
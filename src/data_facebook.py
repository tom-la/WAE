import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import utils
import global_constraint
from process import main_process

def main():
    file = "data/facebook_comments/Training/Features_Variant_1.csv"
    df = pd.read_csv(file, header=None)
    print("Dataset has {} entries and {} features".format(*df.shape))
    X, y = df.loc[:,:52].values, df.loc[:,53].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        # Parameters that we are going to tune.
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': 0.3,
        'subsample': 1,
        'colsample_bytree': 1,
        # Other parameters
        'objective':'reg:linear',
    }

    params['eval_metric'] = "mae"
    num_boost_round = 999

    # model = xgb.train(
    #     params,
    #     dtrain,
    #     num_boost_round=num_boost_round,
    #     evals=[(dtest, "Test")],
    #     early_stopping_rounds=10
    # )

    # print("Best MAE: {:.2f} with {} rounds".format(
    #     model.best_score,
    #     model.best_iteration+1))
    
    main_process(dtrain, dtest, params, 0.01)


if __name__ == "__main__":
    main()
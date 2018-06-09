import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import utils
import global_constraint
from process import main_process
from randomprocess import random_process

def main():
    file = "data/bike/hour2011.csv"
    df = pd.read_csv(file, header=1)
    print("Dataset has {} entries and {} features".format(*df.shape))
    X, y =  df.iloc[:,2:13].values, df.iloc[:,16].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        # Parameters that we are going to tune.
        'max_depth': 3,
        'min_child_weight': 2,
        'eta': 0.6,
        'subsample': 0.2,
        'colsample_bytree': 0.2,
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

    result = main_process(dtrain, dtest, params, 0.1)
    print(result)
    result_random = random_process(dtrain,dtest,result[2])

    print("\t\t\t")
    print(result)
    print(result_random)
if __name__ == "__main__":
    main()

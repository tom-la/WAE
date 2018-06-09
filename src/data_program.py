import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import utils
import global_constraint
from process import main_process
from process_classification import main_process_class
from randomprocess import random_process
from randomprocess_classification import random_process_class

def main():
    file = "data/dota/dota2train.csv"
    df = pd.read_csv(file, header=1)
    print("Dataset has {} entries and {} features".format(*df.shape))
    X, y =  df.iloc[:,1:115].values, df.iloc[:,0].values
    y = [z if z == 1 else 0 for z in y]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        # Parameters that we are going to tune.
        'max_depth': 5,
        'min_child_weight': 1,
        'eta': 0.8,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        # Other parameters
        'objective':'binary:logistic',
    }

    #params['eval_metric'] = "mae"
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

    result = main_process_class(dtrain, dtest, params , 10, y_test )
    #print(result)
    result_random = random_process_class(dtrain,dtest, result[2], y_test)
    #result_random = random_process_class(dtrain,dtest, 100, y_test)

    print("\t\t\t")
    print(result[1])
    print(result_random[1])

    print(result)
    print(result_random)

    print(len(y_test))
if __name__ == "__main__":
    main()

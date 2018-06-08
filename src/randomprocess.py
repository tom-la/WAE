import xgboost as xgb
import utils
from global_constraint import LOWER_BOUND
import random

def random_process(dtrain, dtest, iterations):
    print("Starting hyperparameter tuning with start params:")
    random.seed(a=42)
    min_mae = float("Inf")
    l=0
    for i in range(0, iterations):
        step_params =  {

                'max_depth': 0 + random.randint(0,10) ,
                'min_child_weight': 0 + random.randint(0,10) ,
                'eta': random.uniform(LOWER_BOUND,1),
                'subsample': random.uniform(LOWER_BOUND,1),
                'colsample_bytree': random.uniform(LOWER_BOUND,1),
                'objective': 'reg:linear'
            }

        print(utils.print_params(step_params))
        cv_results = xgb.cv(
            step_params,
            dtrain,
            num_boost_round=10,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = step_params.copy()
            l=l+1
            print(l)




    print("\t")
    print(l)
    print("Found best solution:")
    print(utils.print_params(best_params))
    print("MAE:")
    print(min_mae)

    return (best_params, min_mae)

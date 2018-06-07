import xgboost as xgb
import utils
from global_constraint import LOWER_BOUND
import random

def random_process_class(dtrain, dtest, iterations, y_test):
    print("Starting hyperparameter tuning with start params:")
    random.seed(a=42)
    maxacc = 0
    l=0
    for i in range(0, iterations):
        step_params =  {

                'max_depth': 0 + random.randint(0,10) ,
                'min_child_weight': 0 + random.randint(0,10) ,
                'eta': 0 + random.randint(0,10),
                'subsample': random.uniform(LOWER_BOUND,1),
                'colsample_bytree': random.uniform(LOWER_BOUND,1),
                'objective':'binary:logistic'
            }
        cv_results = xgb.train(
            step_params,
            dtrain,
            num_boost_round=10,
        )
        preds = cv_results.predict(dtest)
        preds = [1 if z > 0.5 else 0 for z in preds]

        #print(preds)
        err = 0

        res = [i for i, j in zip(preds, y_test) if i == j]
        #accuracy = accuracy_score(dtest.label, predictions)
        #print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print(len(res))

        print(100*len(res)/len(preds))

        if len(res) > maxacc:
            maxacc = len(res)
            best_params = step_params.copy()



    print("\t")
    print(l)
    print("Found best solution:")
    print(utils.print_params(best_params))
    print("Random result:")
    print(maxacc)
    print(100*len(res)/len(preds))

    return (best_params, maxacc)

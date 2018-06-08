import xgboost as xgb
import utils
import global_constraint

def main_process_class(dtrain, dtest, params, epsilon, y_test ,stop_value=None):
    print("Starting hyperparameter tuning with start params:")
    print(utils.print_params(params))
    print("With epsilon (stop) value: {}".format(epsilon))
    gradients = utils.get_gradient_list(params, global_constraint.STEP)
    steps = utils.get_possible_steps(params, gradients,[])
    maxacc = 0
    step_mae = 0
    iterations = 0
    best_params = params.copy()
    last_steps = []
    while True:
        last_steps = steps.copy()
        for step_params in steps:
            print(utils.print_params(step_params))
            #bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, , verbose = 2)

            cv_results = xgb.train(
                step_params,
                dtrain,
                num_boost_round=10,
            )
            print(step_mae)
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


        iterations = iterations + 1
        print(iterations)
        if (abs(step_mae - maxacc) < epsilon):
            if(iterations < 500):
                utils.reduce_steps()
                step_mae = maxacc
                steps = utils.get_possible_steps(best_params, gradients, last_steps)
            else:
                break
        else:
            step_mae = maxacc
            print("aaaa")
            steps = utils.get_possible_steps(best_params, gradients, last_steps)

    print("Found best solution:")
    print(utils.print_params(best_params))
    print("MAE:")
    print(maxacc)

    return (params, maxacc, iterations)

import xgboost as xgb
import utils
import global_constraint

def main_process(dtrain, dtest, params, epsilon, stop_value=None):
    print("Starting hyperparameter tuning with start params:")
    print(utils.print_params(params))
    print("With epsilon (stop) value: {}".format(epsilon))
    gradients = utils.get_gradient_list(params, global_constraint.STEP)
    steps = utils.get_possible_steps(params, gradients)
    min_mae = float("Inf")
    step_mae = float("Inf")
    best_params = params.copy()
    while True:
        for step_params in steps:
            print(utils.print_params(step_params))
            cv_results = xgb.cv(
                step_params,
                dtrain,
                num_boost_round=999,
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
        
        if stop_value is not None and min_mae < stop_value:
            break
        
        if (abs(step_mae - min_mae) < epsilon):
            break
        else:
            step_mae = min_mae

    print("Found best solution:")
    print(utils.print_params(best_params))
    print("MAE:")
    print(min_mae)

    return (params, min_mae)
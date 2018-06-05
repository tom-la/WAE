import numpy as np
from global_constraint import STEP, CONSTRAINT

NON_NUMERIC_PARAMS = ['objective', 'eval_metric']

def print_params(params, break_arg=3):
    params_string = ""
    i = 1
    for key, value in params.items():
        params_string = params_string + "{0}: {1}, ".format(key, value)
        if i % break_arg == 0:
            params_string = params_string + "\n"
        i = i + 1
    return params_string

def add_params(params, params2, constraint):
    params1 = params.copy()
    for key, value in params1.items():
        if key in NON_NUMERIC_PARAMS:
            continue
        new_value = params1[key] + params2[key]
        if not (new_value < constraint[key][0] or new_value > constraint[key][1]):
            params1[key] = params1[key] + params2[key]
    return params1

def get_gradient_list(params, steps):
    keys = []
    for key, value in params.items():
        if not key in NON_NUMERIC_PARAMS:
            keys.append(key)
    first_step = steps[keys[0]]
    bounds = [[-first_step], [0], [first_step]]
    for key in keys[1:]:
        step = steps[key]
        # bounds = list(zip(bounds, [-step, 0, step]))
        new_bounds = []
        for b in bounds:
            new_bounds = new_bounds + [b + [-step], b + [0], b + [step]]
        bounds = new_bounds[:]
    # print(bounds)
    # print(len(bounds))
    # bounds = np.unique(np.array(bounds), axis=0).tolist()
    bounds = list(map(lambda b: dict(zip(keys, b)), bounds))
    # bounds = list(map(dict, set(tuple(sorted(d.items())) for d in bounds)))
    # print(len(bounds))
    return bounds

def get_possible_steps(params, gradient_list, last_posibilities):
    possibilities = []
    for g in gradient_list:
        new_params = add_params(params, g, CONSTRAINT)
        if new_params not in possibilities and new_params not in last_posibilities:
            possibilities.append(new_params)
    return possibilities


def main():
    params = {
        # Parameters that we are going to tune.
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': 0.3,
        'subsample': 1,
        'colsample_bytree': 1,
        'objective': 'reg:linear'
    }

    # params2 = {
    #     # Parameters that we are going to tune.
    #     'max_depth': 1,
    #     'min_child_weight': 1,
    #     'eta': -0.1,
    #     'subsample': 1,
    #     'colsample_bytree': -1,
    #     'objective':'reg:linear'
    # }

    # constraint = {
    #     'max_depth': [0, INFINITY],
    #     'min_child_weight': [0, INFINITY],
    #     'eta': [0.0, 1.0],
    #     'subsample': [LOWER_BOUND, 1],
    #     'colsample_bytree': [LOWER_BOUND, 1],
    #     'objective':'reg:linear'
    # }

    # print(print_params(params))
    # add_params(params, params2, constraint)
    print(print_params(params))
    gradient = get_gradient_list(params, STEP)
    # print(gradient)
    print(get_possible_steps(params, gradient, []))

main()

LOWER_BOUND = .000001
INFINITY = 99999

# http://xgboost.readthedocs.io/en/latest/parameter.html
CONSTRAINT = {
    'max_depth': [0, INFINITY],
    'min_child_weight': [0, INFINITY],
    'eta': [0.0, 1.0],
    'subsample': [LOWER_BOUND, 1],
    'colsample_bytree': [LOWER_BOUND, 1],
    'objective':'reg:linear'
}

STEP = {
    'max_depth': 1,
    'min_child_weight': 1,
    'eta': 0.02,
    'subsample': 0.02,
    'colsample_bytree': 0.04,
    'objective':'reg:linear'
}

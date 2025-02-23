models = {'Linear regression': LinearRegression(),
          'Ridge': Ridge(),
          'Lasso': Lasso(),
          'XGBoost': XGBoost()}

results = []

for model in models.values():
    kf = KFold
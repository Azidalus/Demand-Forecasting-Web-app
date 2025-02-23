from data_ingestion.py import

models = {'Linear regression': LinearRegression(),
          'Ridge': Ridge(),
          'Lasso': Lasso(),
          'XGBoost': XGBoost()}

results = []

for model in models.values():
    kf = KFold
    cv_results = 
    results.append(cv_results)

for name, model in models.items():
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("{} Test Score Accuracy: {}".format(name, test_score))
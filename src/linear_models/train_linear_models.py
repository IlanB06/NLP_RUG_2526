from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def train_linear_models(X_train, y_train) -> dict:
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)


    return {
        "logistic_regression": lr_model,
        "svm": svm_model
    }
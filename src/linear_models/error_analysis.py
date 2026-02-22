from data.preprocessing import preprocess_ag_news
from src.linear_models.train_linear_models import train_linear_models

def collect_misclassified_svm(n=20):
    data = preprocess_ag_news()
    label_names = data["label_names"]
    models = train_linear_models(data["train"]["X"], data["train"]["y"])
    svm = models["svm"]

    X_test = data["test"]["X"]
    y_test = list(data["test"]["y"])
    raw_test = list(data["test"]["raw_text"])

    y_pred = svm.predict(X_test)

    print(f"Collecting first 20 misclassified examples from TEST:\n")

    count = 0
    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            print("=" * 80)
            print(f"Index: {i}")
            print(f"TRUE: {label_names[y_test[i]]}    PRED: {label_names[int(y_pred[i])]}")
            print(f"TEXT: {raw_test[i]}")
            count += 1
            if count == n:
                break

    print("\nDone.")
    print(f"Found {count} misclassified examples.")

if __name__ == "__main__":
    collect_misclassified_svm(n=20)

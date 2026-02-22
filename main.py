from src.linear_models.eval_linear_models import evaluate_linear_models


def main():
    evaluate_linear_models()
    collect_misclassified_svm(n=20)


if __name__ == "__main__":
    main()

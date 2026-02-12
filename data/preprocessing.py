from sklearn.feature_extraction.text import TfidfVectorizer
from data.load_data import load_ag_news

def preprocess_ag_news() -> dict:
    datasets = load_ag_news()

    vectorizer = TfidfVectorizer()
    vectorizer.fit(datasets["train"]["text"])

    vectorized_datasets = {}

    for dataset in ["train", "dev", "test"]:
        vectorized_datasets[dataset] = {}
        vectorized_datasets[dataset]["text"] = vectorizer.transform(datasets[dataset]["text"])
        vectorized_datasets[dataset]["label"] = datasets[dataset]["label"]


    return vectorized_datasets

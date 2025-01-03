# import Libs
import sklearn.metrics as metrics
from sklearn.datasets import fetch_20newsgroups

from LLM_init import IMDB_BinaryClassifier, NewsGroup_MultiClassClassifier, Emotions_MultiClassClassifier
import pandas as pd
from tqdm import tqdm

# Base class to evaluate classifier objects and calculate recall, precision, F1 scores
class BaseEvaluator:

    def __init__(self, classifier, dataset, averageType='binary'):
        self.classifier = classifier
        self.dataset = dataset
        self.averageType = averageType

    def calculate_metrics(self, groundTruths, candidatePreds):

        precision = metrics.precision_score(groundTruths, candidatePreds, average=self.averageType)
        recall = metrics.recall_score(groundTruths, candidatePreds, average=self.averageType)
        f1 = metrics.f1_score(groundTruths, candidatePreds, average=self.averageType)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def evaluate(self, percentOfDataset=0.66):

        # init dataset subset for eval
        evalset_len = int(len(self.dataset) * (percentOfDataset / 100))
        # init ground truths, candidate preds
        groundTruths = []
        candidatePreds = []

        for _, row in tqdm(self.dataset.head(evalset_len).iterrows(), total=evalset_len, desc="Eval progress ..."):
            try:
                candidatePreds.append(self.classifier.classify(row['text']))
                groundTruths.append(row['label'])
            except Exception as e:
                print(f"received exception {e}, moving to next data point!")

        return self.calculate_metrics(groundTruths=groundTruths, candidatePreds=candidatePreds)


# IMDB Binary classifier evaluator
# dataset source: https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis?resource=download
class IMDBBinaryClassifier_Evaluator(BaseEvaluator):

    def __init__(self):
        classifier = IMDB_BinaryClassifier()
        dataset = pd.read_csv("./Dataset/movie.csv")

        BaseEvaluator.__init__(self, classifier, dataset)


# 20 Newsgroups Multi-Class classifier evaluator
# dataset source: https://scikit-learn.org/dev/modules/generated/sklearn.datasets.fetch_20newsgroups.html
class NewsGroupMultiClassClassifier_Evaluator(BaseEvaluator):

    def __init__(self):
        classifier = NewsGroup_MultiClassClassifier()
        raw_set = fetch_20newsgroups(subset='test')
        dataset = pd.DataFrame({"text":raw_set.data, "label":raw_set.target})

        BaseEvaluator.__init__(self, classifier, dataset, averageType='micro')


# Emotions Multi-Class classifier evaluator
# dataset source: https://huggingface.co/datasets/dair-ai/emotion
# https://github.com/dair-ai/emotion_dataset
class EmotionsMultiClassClassifier_Evaluator(BaseEvaluator):

    def __init__(self):
        classifier = Emotions_MultiClassClassifier()
        raw_set = pd.read_pickle("./Dataset/emotions.pkl")
        dataset = raw_set.rename(columns={'emotions': 'label'})

        BaseEvaluator.__init__(self, classifier, dataset, averageType='micro')

if __name__ == "__main__":

    # Evaluating only 50 entries per datasets to get evals quickly
    # percentOfDataset can be set to 100 to eval all entries for a given dataset

    print("Evaluate Binary Classifier on IMDB dataset")
    eval_obj = IMDBBinaryClassifier_Evaluator()
    res = eval_obj.evaluate(percentOfDataset=0.125)
    print(f"Eval result: {res}")

    print("Evaluate Multi-Class Classifier on 20 News Group dataset")
    eval_obj = NewsGroupMultiClassClassifier_Evaluator()
    res = eval_obj.evaluate(percentOfDataset=0.663)
    print(f"Eval result: {res}")

    print("Evaluate Multi-Class Classifier on Emotions dataset")
    eval_obj = EmotionsMultiClassClassifier_Evaluator()
    res = eval_obj.evaluate(percentOfDataset=0.0119)
    print(f"Eval result: {res}")
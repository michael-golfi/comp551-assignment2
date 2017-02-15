import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

def main():
    FILENAME = "data/trial1/train_input.csv"
    CATEGORY = "data/trial1/train_output.csv"
    
    train_input_X = pd.read_csv(FILENAME, usecols=["conversation"])
    train_input_Y = pd.read_csv(CATEGORY, usecols=["category"])
    train_input_XY = pd.concat([train_input_X, train_input_Y], axis=1)

    run_pipeline(naive_bayes(), train_input_XY)
    
    #run_pipeline(svm(), train_input_XY)
    
    #run_pipeline(decision_tree(), train_input_XY)

def run_pipeline(pipeline, data):
    kf = KFold(shuffle=True, n_splits=3)
    for train, test in kf.split(data):
        conversationsX = data.iloc[train]["conversation"].values
        conversationsY = data.iloc[train]["category"].values

        testX = data.iloc[test]["conversation"].values
        testY = data.iloc[test]["category"].values

        pipeline.fit(conversationsX, conversationsY)
        results = pipeline.predict(testX)

        report = classification_report(testY, results)    
        print report

def naive_bayes():
    print "Run Naive Bayes"
    return Pipeline([
        ('count',  CountVectorizer(ngram_range=(1, 2))),
        ('tfidf',  TfidfTransformer()),
        ('classify',  MultinomialNB())
    ])

def svm():
    print "Run SVM"
    return Pipeline([
        ('count',  CountVectorizer(ngram_range=(1, 2))),
        ('tfidf',  TfidfTransformer()),
        ('classify',  SVC(kernel='linear'))
    ])

def decision_tree():
    print "Run Decision Tree"
    return Pipeline([
        ('count',  CountVectorizer(ngram_range=(1, 2))),
        ('tfidf',  TfidfTransformer()),
        ('classify',  DecisionTreeClassifier())
    ])

if __name__ == "__main__":
    main()
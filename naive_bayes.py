"""
COMP 551 - Assignment 2
Text Classification

Authors:
    Michael Golfi <michael.golfi@mail.mcgill.ca>
    Shruti Bhanderi <shruti.bhanderi@mail.mcgill.ca>
    Zahra Khambaty <zahra.khambaty@mail.mcgill.ca>
"""
import pandas as pd
import math
from collections import Counter
import numpy as np
import os.path

"""
categories = ["hockey", "movies", "nba", "news", "nfl", "politics", "soccer", "worldnews"]
total_word_count = 0
totalsPerCategory = {}
vocabulary = []"""

def word_count(df, column_name):
    words = df[column_name].str.split().tolist()
    return pd.DataFrame(Counter(pd.DataFrame(words).stack()).items(), columns=["word", "count"])

def probabilityIntersect(df, category, word):
    cat = df.loc[category]
    return cat[cat["word"] == word]

def probability(df, word):
    return df[df["word"] == word]

pd.DataFrame.word_count = word_count
pd.DataFrame.probabilityIntersect = probabilityIntersect
pd.DataFrame.probability = probability

def validate(model, testSet):
    accuratelyFound = 0

    likelihood = dict([ (topic, np.float64(1.0)) for topic in model.keys() ])
    for index, instance in testSet.iterrows():        
        
        for topic in likelihood.keys():
        #    print "Testing: %s" % instance["conversation"]
            probability = np.float64(1.0)
            for word in instance["conversation"].split():
                
                probability *= np.float64(model[topic][word]) if word in model[topic] else np.float64(1.0)                
                likelihood[topic] = probability
        
        max_label = min(likelihood.iterkeys(), key=(lambda key: likelihood[key]))
        #print instance["category"], max_label, likelihood

        if instance["category"] == max_label:
            accuratelyFound += 1

    return accuratelyFound/float(len(testSet)) * 100.0

def calculate_total_vocabulary(train):
    totalFrequencies = train.word_count("conversation")
    totalFrequencies = totalFrequencies[totalFrequencies["count"] > 2]
    return (totalFrequencies, totalFrequencies["count"].sum())

def calculate_probabilities(vocabulary, total_vocab_count, categoryWeights, categories):
    probabilities = {}

    vocab = vocabulary.set_index("word")["count"].to_dict()
    catWeights = categoryWeights.set_index("category")["count"].to_dict()
    totalCategories = categoryWeights["count"].sum()
    
    ## Need to use P(Category | Word) = P(Word | Category) * P(Category) / P(Word)
    for (name, category) in categories:
        words = {}

        categoryWordDf = category.word_count("conversation")
        categoryWords = categoryWordDf.set_index("word")["count"].to_dict()
        totalWordsPerCategory = np.float64(categoryWordDf["count"].sum())
        
        for word in vocab.keys():

            if word in categoryWords:
                wordPerCategory = np.float64(categoryWords[word])
                
                categoryWeight = np.float64(catWeights[name])
                wordOccurenceVocab = vocab[word]

                pWC = np.float64(1.0 + wordPerCategory / totalWordsPerCategory)
                pC = np.float64(categoryWeight / totalWordsPerCategory)
                pW = np.float64(1.0 + wordOccurenceVocab / total_vocab_count)
                words[word] = np.float64(pWC * pC / pW)

        probabilities[name] = words

    return probabilities

def main():
    FILENAME = "data/train_input_edited.csv"
    CATEGORY = "data/train_output.csv"
    COUNT_FILE = "data/train_input_counts.csv"
    TRAINING_THRESHOLD = 0.7

    train_input_X = pd.read_csv(FILENAME, usecols=["conversation"])
    train_input_Y = pd.read_csv(CATEGORY, usecols=["category"])
    train_input_XY = pd.concat([train_input_X, train_input_Y], axis=1)

    print "Splitting into training and testing"
    cutoff = np.random.rand(len(train_input_XY)) < TRAINING_THRESHOLD
    train = train_input_XY[cutoff]
    test = train_input_XY[~cutoff]

    print "Count all distinct words in vocabulary"
    vocab, vocab_word_count = calculate_total_vocabulary(train)
    print "Found %d words in vocabulary" % vocab_word_count

    print "Count category occurences"
    categoryWeights = pd.DataFrame(Counter(pd.DataFrame(train_input_Y).stack()).items(), columns=["category", "count"])

    print "Calculating probabilities for all classes"
    probabilities = calculate_probabilities(vocab, vocab_word_count, categoryWeights, train.groupby(["category"]))

    print "Validating Dataset"
    accuracy = validate(probabilities, test)
    print "Found %d accuracy" % accuracy

if __name__ == "__main__":
    main()
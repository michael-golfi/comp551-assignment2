"""
COMP 551 - Assignment 2
Text Classification

Authors:
    Michael Golfi <michael.golfi@mail.mcgill.ca>
    Shruti Bhanderi <shruti.bhanderi@mail.mcgill.ca>
    Zahra Khambaty <zahra.khambaty@mail.mcgill.ca>
"""
import pandas as pd
from math import log
from collections import Counter
import numpy as np
import os.path

def word_count(df, column_name):
    words = df[column_name].str.split().tolist()
    return pd.DataFrame(Counter(pd.DataFrame(words).stack()).items(), columns=["word", "count"])

def classify(model, testSet, totalWordCount, totalWordsInCategories):
    found = np.float(0.0)
    alpha = 1.0
    
    predictions = []
    for i, instance in testSet.iterrows():
        likelihood = {}
        for category in model.keys():
            pWC = 0

            for word in instance["conversation"].split():

                if word in model[category]:
                    pWC += log( (model[category][word] + alpha ) / ( len(model[category]) + totalWordCount + alpha ), 2)
                else:
                    pWC += log(alpha / ( len(model[category]) + totalWordCount + alpha ), 2 )

            #print "Category: %s length: %d, WordsInCat: %s, pWC: %f" % (category, len(model[category]), totalWordsInCategories, pWC)
            likelihood[category] = log(np.float64(len(model[category])) / np.float64(totalWordsInCategories), 2) + pWC
        
        max_label = max(likelihood.iterkeys(), key=(lambda key: likelihood[key]))
        #print instance["category"], max_label, likelihood
        if instance["category"] == max_label:
            found += 1

    return found/float(len(testSet)) * 100.0

def calculate_probabilities(vocabulary, total_vocab_count, categoryWeights, categories):
    probabilities = {}

    vocab = vocabulary.set_index("word")["count"].to_dict()
    catWeights = categoryWeights.set_index("category")["count"].to_dict()
    totalCategories = categoryWeights["count"].sum()
    
    ## Need to use P(Category | Word) = P(Word | Category) * P(Category) / P(Word)
    for (name, category) in categories:
        words = {}

        categoryWordDf = word_count(category, "conversation")
        categoryWords = categoryWordDf.set_index("word")["count"].to_dict()
        totalWordsPerCategory = np.float64(categoryWordDf["count"].sum())
        
        for word in vocab.keys():

            if word in categoryWords:
                wordPerCategory = np.float64(categoryWords[word])
                
                categoryWeight = np.float64(catWeights[name])
                wordOccurenceVocab = np.float64(vocab[word])

                #pWC = np.float64(1.0 + wordPerCategory / totalWordsPerCategory)
                #pC = np.float64(categoryWeight / totalWordsPerCategory)
                #pW = np.float64(1.0 + wordOccurenceVocab / total_vocab_count)
                #words[word] = np.float64((pWC * pC) / pW)
                alpha = 0.0005
                words[word] = np.float64( ( wordPerCategory + alpha ) / (totalWordsPerCategory + alpha * total_vocab_count ) )
                
        probabilities[name] = words

    return probabilities

def main():
    FILENAME = "data/train_input.csv"
    CATEGORYCOUNTS = "data/category_count.csv"
    COUNT_FILE = "data/train_input_count.csv"
    CATEGORY = "data/train_output.csv"
    
    TRAINING_THRESHOLD = 0.8

    train_input_X = pd.read_csv(FILENAME, usecols=["conversation"])
    train_input_Y = pd.read_csv(CATEGORY, usecols=["category"])
    train_input_XY = pd.concat([train_input_X, train_input_Y], axis=1)

    print "Splitting into training and testing"
    cutoff = np.random.rand(len(train_input_XY)) < TRAINING_THRESHOLD
    train = train_input_XY[cutoff]
    test = train_input_XY[~cutoff]

    print "Count all distinct words in vocabulary"
    vocab = pd.read_csv(COUNT_FILE)
    vocab_word_count = vocab["count"].sum()
    print "Found %d words in vocabulary" % vocab_word_count

    print "Count category occurences"
    categoryWeights = pd.read_csv(CATEGORYCOUNTS)

    print "Calculating probabilities for all classes"
    model = {}
    instancesInAllModels = 0
    for (name, category) in train.groupby(["category"]):
        model[name] = word_count(category, "conversation").set_index("word")["count"].to_dict()
        instancesInAllModels += len(model[name])
    
    print instancesInAllModels

    print "Classifying..."
    acc = classify(model, test, vocab_word_count, instancesInAllModels)
    print acc

    #probabilities = calculate_probabilities(vocab, vocab_word_count, categoryWeights, train.groupby(["category"]))

    #print "Validating Dataset"
    #accuracy = validate(probabilities, test)
    #print "Found %f accuracy" % accuracy

if __name__ == "__main__":
    main()
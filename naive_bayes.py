"""
COMP 551 - Assignment 2
Text Classification

Authors:
    Michael Golfi <michael.golfi@mail.mcgill.ca>
    Shruti Bhanderi <shruti.bhanderi@mail.mcgill.ca>
    Zahra Khambaty <zahra.khambaty@mail.mcgill.ca>
"""
import pandas as pd
from collections import Counter
import numpy as np
import os.path

def word_count(df, column_name):
    words = df[column_name].str.split().tolist()
    return pd.DataFrame(Counter(pd.DataFrame(words).stack()).items(), columns=["word", "count"])

def probability(df, category, word):
    cat = df.loc[category]
    return cat[cat["word"] == word]

def calculate_frequencies(group):
    group = group.word_count("conversation")
    total = group["count"].sum()
    group["frequency"] = group["count"].map(lambda x: np.float64(x) / np.float64(total))
    return group

pd.DataFrame.word_count = word_count
pd.DataFrame.probability = probability

FILENAME = "data/train_input_edited.csv"
CATEGORY = "data/train_output.csv"
COUNT_FILE = "data/train_input_counts.csv"

df = pd.concat([pd.read_csv(FILENAME, usecols=["conversation"]), pd.read_csv(CATEGORY, usecols=["category"])], axis=1)
topicIntersectWord = df.groupby(["category"]).apply(calculate_frequencies)

print topicIntersectWord.probability("nba", "jump")
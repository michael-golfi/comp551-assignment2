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

FILENAME = "data/train_input_edited.csv"
CATEGORY = "data/train_output.csv"
COUNT_FILE = "data/train_input_counts.csv"

posts = pd.read_csv(FILENAME)
categories = pd.read_csv(CATEGORY)

"""if not os.path.isfile(COUNT_FILE):
    word_occurence = pd.DataFrame(posts.conversation.str.split().tolist()).stack().value_counts()
    word_occurence.to_csv("data/train_input_counts.csv", columns=["word", "count"])
else:
    word_occurence = pd.read_csv(COUNT_FILE)"""

postCategories = pd.concat([posts, categories], axis=1)[["conversation", "category"]]
categoryGroups = postCategories.groupby(["category"])

probabilities = {}

## Change this to transform...
for (k,v) in categoryGroups:
    
    # Get unique word count in categories
    words = v.conversation.str.split().tolist()
    word_occurence = pd.DataFrame(Counter(pd.DataFrame(words).stack()).items(), columns=["word", "count"])
    word_occurence = word_occurence[word_occurence["count"] > 2]
    total_words_category = word_occurence["count"].sum()

    word_occurence["frequency"] = word_occurence["count"].map(lambda x: np.float64(x) / np.float64(total_words_category))
    probabilities[k] = word_occurence
    
print probabilities

def probability(groups, group_name, word):
    category = groups[group_name]
    return category[category["word"] == word].frequency

print probability(probabilities, "nba", "jump")
import pandas as pd
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Assumes that nltk stopwords and lemmatizer are downloaded.

FILENAME = "project data/train_input.csv"
CATEGORY = "project data/train_output.csv"
OUTPUT = "data/train_input.csv"
OUTPUT_OCC = "data/train_input_count.csv"

stopWords = stopwords.words('english')
lemma = WordNetLemmatizer()

filter = lambda x: x not in stopWords and len(x) > 3

def pre_process(words):
    """
    Lowercases all words
    Removes punctuation
    Removes everything except letters
    Removes extra spaces
    Removes stopwords
    Lemmatizes all words
    """

    remove_punctuation = re.sub('<[^>]*>|[.,\/#!$%\^&\*;:{}=\-_`~()"\?\']|@\w+', "", words.lower()).strip()
    keep_only_letters = re.sub('[^a-zA-Z]', ' ', remove_punctuation)
    strip_whitespace = re.sub(' +', " ", keep_only_letters)
    remove_stopwords = ' '.join([lemma.lemmatize(word) for word in strip_whitespace.split() if filter(word)])
    return remove_stopwords

def word_count(df, column_name):
    words = df[column_name].str.split().tolist()
    return pd.DataFrame(Counter(pd.DataFrame(words).stack()).items(), columns=["word", "count"])

df = pd.read_csv(FILENAME)[["id", "conversation"]]
df["conversation"] = df["conversation"].map(pre_process)
df.to_csv(OUTPUT)

vocabularyOcc = word_count(df, "conversation")
print vocabularyOcc
vocabularyOcc.to_csv(OUTPUT_OCC)

categories = pd.read_csv(CATEGORY)
categoryCount = word_count(categories, "category")
categoryCount.to_csv("data/category_count.csv", columns=["category", "count"], index=False)
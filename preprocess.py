import pandas as pd
import re
import enchant
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

FILENAME = "data/train_input.csv"
OUTPUT = "data/train_input_edited.csv"

stopWords = stopwords.words('english')
lemma = WordNetLemmatizer()

filter = lambda x: x not in stopWords and len(x) > 1

def pre_process(words):
    """
    Lowercases all words
    Removes punctuation
    Removes extra spaces
    Removes stopwords
    Lemmatizes all words
    """

    remove_punctuation = re.sub('<[^>]*>|[.,\/#!$%\^&\*;:{}=\-_`~()"\?\']|@\w+', "", words.lower()).strip()
    strip_whitespace = re.sub(' +', " ", remove_punctuation)
    remove_stopwords = ' '.join([word for word in strip_whitespace.split() if filter(word)])
    return lemma.lemmatize(remove_stopwords)


df = pd.read_csv(FILENAME)
df["conversation"] = df["conversation"].map(pre_process)
df.to_csv(OUTPUT)
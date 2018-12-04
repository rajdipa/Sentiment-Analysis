import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer as rt

# read in the csv into a dataframe object
data = pd.read_csv("train.csv") # saves into a DataFrame object
phrases = data["Phrase"]
# print first 5 rows of data
#print(data.head())
# print the sentiment column in two different ways
#print(data.loc[:, "Sentiment"])
#print(data[["Sentiment"]])


matrix = CountVectorizer(max_features=1000)
X = matrix.fit_transform(data).toarray()
print(X)
# Output: number of times each unique word appears in each phrase
# [[0 1 0 0]
#  [0 0 1 0]
#  [1 0 0 0]
#  [0 0 0 1]]

# Define our RegExp tokenizer to keep only alphanumeric characters.
tokenizer = rt(r'\w+')
phrases = (tokenizer.tokenize(phrase) for phrase in phrases)
list = []
for x in phrases:
    list.append(x)
print(list)


# functions
def read_data(fileName):
    d = pd.read_csv(fileName)
    return d

def clean_punct(dataset):
    d = re.sub('[^A-Za-z]', '', dataset)
    return d

def clean_caps(dataset):
    d = dataset.lower()
    return d

def clean_stem(dataset):
    stemmer = PorterStemmer()

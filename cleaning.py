import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer as rt

from nltk.corpus import stopwords

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
# Make all lowercase.
phrases = (phrase.lower() for phrase in phrases)
phrases = (tokenizer.tokenize(phrase) for phrase in phrases)
list = []
for x in phrases:
    list.append(x)
print(list) # a list of lists of strings

# Create a new Stopwords list object, built-in to nltk. Filter out these words.
# stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stop_words = list(stopwords.words('english'))
custom_words = ['genre', 'film']
stop_words = set(stop_words).union(set(set(custom_words)))

result = []
for item in list:
    temp = []
    for i in item:
        if not i in stop_words:
            temp.append(i)
    result.append(temp)
print(result)


# functions
# def read_data(fileName):
#     d = pd.read_csv(fileName)
#     return d
#
# def clean_punct(dataset):
#     d = re.sub('[^A-Za-z]', '', dataset)
#     return d
#
# def clean_caps(dataset):
#     d = dataset.lower()
#     return d
#
# def clean_stem(dataset):
#     stemmer = PorterStemmer()

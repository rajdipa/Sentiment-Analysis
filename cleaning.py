# Tokenize words, clean data, remove noun words with no sentiment using NLTK's word packages.
# Save changes to data for each step - tokenize, stem, remove stopwords - into txt files, to see differences with each cleaning.
# Check the develop branch for comparisons between txt files.

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('words')

import pandas as pd
import re

from nltk.tokenize import RegexpTokenizer as rt
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import pickle as rick

# Read in the csv into a dataframe object
data = pd.read_csv("train.csv") # saves into a DataFrame object
phrases = data["Phrase"]
labels = data["Sentiment"]


# Creates a n by m word matrix of n phrases and m unique words.
# matrix = CountVectorizer(max_features=1000)
# X = matrix.fit_transform(data).toarray()
# print(X)

# Output: number of times each unique word appears in each phrase
# [[0 1 0 0]
#  [0 0 1 0]
#  [1 0 0 0]
#  [0 0 0 1]]

# Initialize our RegExp tokenizer to filter out non-alphanumeric characters.
tokenizer = rt(r'\w+')
# Make all lowercase.
phrases = (phrase.lower() for phrase in phrases)
phrases = (tokenizer.tokenize(phrase) for phrase in phrases)
cleaned = []
for phrase in phrases:
    cleaned.append(phrase)
#print(cleaned) # a list of lists of strings
# with open('tokenized.txt', 'w') as f:
#     for item in cleaned:
#         f.write("%s\n" % item)
# f.close()

# Initialize lemmatizer, which stems words.
# stemmer = WordNetLemmatizer()
# stemmed = []
# for phrase in cleaned:
#     temp = []
#     for word in phrase:
#         temp.append(stemmer.lemmatize(word))
#     stemmed.append(temp)
#print(stemmed) # still a list of lists of strings
# with open('stemmed.txt', 'w') as f:
#     for item in stemmed:
#         f.write("%s\n" % item)
# f.close()

# Initialize a Stopwords list, imported from nltk. Remove articles, conjunctions, prepositions.
# stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stop_words = list(stopwords.words('english'))
# Extra words to filter out:
custom_words = ['genre', 'film', 'movie']
stop_words = set(stop_words).union(set(custom_words))

good_words = set(list(words.words()))

# First cleaning to only remove stopwords
# result = []
# for phrase in cleaned:
#     temp = []
#     for word in phrase:
#         if not word in stop_words:
#             temp.append(word)
#     result.append(temp)
#print(result)
# with open('firstclean.txt', 'w') as f:
#     for item in result:
#         f.write("%s\n" % item)
# f.close()

# Second cleaning to remove words not in goodwords
result = []
for phrase in cleaned:
    temp = []
    for word in phrase:
        if word in good_words and not word in stop_words:
            temp.append(word)
    result.append(temp)
print(result)
# with open('secondclean.txt', 'w') as f:
#     for item in result:
#         f.write("%s\n" % item)
# f.close()

# Is using stopwords useful?
# unix $ diff firstclean.txt secondclean.txt


# Create pickle file
def generate_cleaned_file(instances, labels):
    col = ["Phrases", "Sentiment"]
    i = range(len((instances)))
    dataframe = pd.DataFrame(columns = col, index = i)
    dataframe["Phrases"] = instances
    dataframe["Sentiment"] = labels
    pkfile = open('preprocessed_train_data.pkl', 'wb')
    rick.dump(dataframe, pkfile)
    pkfile.close()

# Read pickle file
def open_cleaned_file(fileName):
    pkl = open(fileName, 'rb')
    result = rick.load(pkl)
    pkl.close()
    return result

# Print contents of the pickle file
generate_cleaned_file(result, labels)
dataset = open_cleaned_file('preprocessed_train_data.pkl')
print(type(dataset))
print(dataset)


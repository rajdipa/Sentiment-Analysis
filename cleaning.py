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
# from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as rick

# FUNCTIONS---------------------------------------

# Save contents of fname csv into a DataFrame object
# Choose the column you wish you return
def read_data(fName, colName):
    data = pd.read_csv(fName)
    return data[colName]

# Output result of cleaning into a txt file, by the name of fname
# resultlist is a list of lists, the items in the inner list are strings
def print_result(fName, resultlist):
    with open(fName, 'w') as f:
        for l in result:
            f.write("%s\n" % l)
    f.close()

# Dump the preprocessed data into a pickle file
def generate_cleaned_file(instances, labels):
    col = ["Phrases", "Sentiment"]
    i = range(len((instances)))
    dataframe = pd.DataFrame(columns = col, index = i)
    dataframe["Phrases"] = instances
    dataframe["Sentiment"] = labels
    pkfile = open('preprocessed_train_data.pkl', 'wb')
    rick.dump(dataframe, pkfile)
    pkfile.close()

# Open the pickle file
def open_cleaned_file(fileName):
    pkl = open(fileName, 'rb')
    result = rick.load(pkl)
    pkl.close()
    return result

# Tokenize and filter out non-alphanumeric characters.
def tokenize_data(d):
    # Initialize RegExp tokenizer.
    tokenizer = rt(r'\w+')
    # Make all words lowercase.
    d = (phrase.lower() for phrase in d)
    d = (tokenizer.tokenize(phrase) for phrase in d)
    tokenized = []
    for phrase in d:
        tokenized.append(phrase)
    # print_result('cleaned.txt', result)
    return tokenized

# Stem words.
# Ex: [swim, swims, swimming] -> [swim]
def stem_data(d):
    # Initialize lemmatizer: "stem" similar words.
    stemmer = WordNetLemmatizer()
    stemmed = []
    for phrase in d:
        temp = []
        for word in phrase:
            temp.append(stemmer.lemmatize(word))
        stemmed.append(temp)
    #print_result('stemmed.txt', result)
    return stemmed

# Filter out prepositions, conjunctions, keep adjectives, nouns.
def filter_data(d):
    # Initialize stopwords list: remove articles, conjunctions, prepositions, pronouns.
    # stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stop_words = list(stopwords.words('english'))
    # Additional non-sentimental words to filter out
    custom_words = ['genre', 'film', 'movie']
    stop_words = set(stop_words).union(set(custom_words))
    # Second cleaning
    # Initialize dictionary of useful words from nltk corpus
    good_words = set(list(words.words()))
    result = []
    for phrase in d:
        temp = []
        for word in phrase:
            if word in good_words and not word in stop_words:
                temp.append(word)
        result.append(temp)
    # print_result('secondclean.txt', result)
    return result




# MAIN------------------------------------

# Get data
phrases = read_data("train.csv", "Phrase")
labels = read_data("train.csv", "Sentiment")
# Tokenize and clean
cleaned = tokenize_data(phrases)
# Deemed unhelpful, so we commented this part out:
# cleaned = stem_data(cleaned)
# Filter out non-sentimental words
result = filter_data(cleaned)

# Generate the pickle file
generate_cleaned_file(result, labels)
dataset = open_cleaned_file('preprocessed_train_data.pkl')
print(type(dataset))
print(dataset)
print(len(result)) # 109,242

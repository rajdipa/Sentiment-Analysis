# naive bayes from https://opensourceforu.com/2016/12/analysing-sentiments-nltk/
import csv
import re
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

def readcsv():
    with open('train.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        cnt = 0
        useless_words = ['and', 'the', 'of', '']
        for row in spamreader:
            # skip first line of train file (names)
            if cnt > 0:
                # clean: remove punctuation
                phrase = re.sub(r'[^\w\s]','',row[2])
                label = row[3]
                train.append((phrase, label))
            cnt +=1
            if cnt > 50: break

if __name__ == "__main__":
    # Step 1 load in data
    train = []
    readcsv()
    #print(train)

    # Step 2 clean: convert to lowercase
    # passage is a tuple in the train dictionary
    dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
    #print(dictionary)

    # Step 3
    t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
    print(t)

    # Step 4 – the classifier is trained with sample data
    classifier = nltk.NaiveBayesClassifier.train(t)

    test_data = "that was terrible i hate it"
    test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}

    print (classifier.classify(test_data_features))

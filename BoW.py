# https://pythonspot.com/python-sentiment-analysis/
import csv
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

def readcsv():
    with open('train.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        cnt = 0
        for row in spamreader:
            if cnt > 0: 
                string = row[2]
                label = row[3]
                splitstr = string.split(' ')
                for i in splitstr:
                    bag[label].append(i)
            cnt +=1
            if cnt == 10: break

def word_feats(words):
    return dict([(word, True) for word in words])
 

if __name__ == "__main__":
    bag = {'0':[],'1':[],'2':[],'3':[],'4':[]}
    readcsv()
    negative_features = [(word_feats(neg), 'neg') for neg in bag['0']]
    sw_negative_features = [(word_feats(swneg), 'swneg') for swneg in bag['1']]
    neutral_features = [(word_feats(neu), 'neu') for neu in bag['2']]
    sw_positive_features = [(word_feats(swpos), 'swpos') for swpos in bag['3']]
    positive_features = [(word_feats(pos), 'pos') for pos in bag['4']]
    
    train_set = negative_features + sw_negative_features +positive_features + sw_positive_features + neutral_features
 
    classifier = NaiveBayesClassifier.train(train_set) 
    
        # Predict
    neg = 0
    swneg = 0
    swpos = 0
    pos = 0
    sentence = "Awesome movie, I liked it"
    sentence = sentence.lower()
    words = sentence.split(' ')
    for word in words:
        classResult = classifier.classify( word_feats(word))
        if classResult == 'neg':
            neg = neg + 1
        if classResult == 'swneg':
            swneg = swneg + 1
        if classResult == 'swpos':
            swpos = swpos + 1    
        if classResult == 'pos':
            pos = pos + 1
     
    print('Positive: ' + str(float(pos)/len(words)))
    print('SWPositive: ' + str(float(swpos)/len(words)))
    print('SWNegative: ' + str(float(swneg)/len(words)))
    print('Negative: ' + str(float(neg)/len(words)))
    
    
    
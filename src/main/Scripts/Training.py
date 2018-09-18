import re
import csv
import pprint
import nltk.classify
import pickle
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC

def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)

def processMessage(msg):
    msg = msg.lower()
    msg = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',msg)
    msg = re.sub('@[^\s]+','USER',msg)    
    msg = re.sub('[\s]+', ' ', msg)
    msg = re.sub(r'#([^\s]+)', r'\1', msg)
    msg = msg.strip('\'"')
    return msg

def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('USER')
    stopWords.append('URL')
    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

def getFeatureVector(msg, stopWords):
    featureVector = []
    words = msg.split()
    for w in words:
        w = replaceTwoOrMore(w) 
        w = w.strip('\'"?,.')
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector    


def extract_features(msg):
    msg_words = set(msg)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in msg_words)
    return features


dircetory = "C:\\Users\\User\\AML-Project-1\\src\\main\\"

#Read the messages one by one and process it
inpMessages = csv.reader(open(dircetory + 'Resources\\sms_spam_train.csv', 'r', encoding = "cp850"))
print(inpMessages)
stopWords = getStopWordList(dircetory + 'Resources\\stopwords.txt')
count = 0
featureList = []
messages = []

for row in inpMessages:
    sentiment = row[0]
    message = row[1]
    processMessage = processMessage(message)
    featureVector = getFeatureVector(processMessage, stopWords)
    featureList.extend(featureVector)
    messages.append((featureVector, sentiment))

featureList = list(set(featureList))

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, messages)

print("Train the Naive Bayes classifier")
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
print("Trained NaiveBayes_Classifier")

filename = 'NaiveBayes_Classifier.sav'
pickle.dump(NBClassifier, open(dircetory + "Output\\Models\\" + filename, 'wb'))


print("Training SVC_classifier")
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("Trained SVC_classifier")

filename1 = 'SVC_classifier.sav'
pickle.dump(SVC_classifier, open(dircetory + "Output\\Models\\" + filename1, 'wb'))


print("Training Logisitic Regression")
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("Trained Logisitic Regression")

filename3 = 'LogisticRegression_classifier.sav'
pickle.dump(LogisticRegression_classifier, open(dircetory + "Output\\Models\\" + filename3, 'wb'))


print("Training MNB_classifier")
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Trained MNB_classifier")

filename4 = 'MNB_classifier.sav'
pickle.dump(MNB_classifier, open(dircetory + "Output\\Models\\" + filename4, 'wb'))


print("Training SGDClassifier_classifier")
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("Trained SGDClassifier_classifier")

filename5 = 'SGDClassifier_classifier.sav'
pickle.dump(SGDClassifier_classifier, open(dircetory + "Output\\Models\\" + filename5, 'wb'))


print("Training LinearSVC_classifier")
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("Trained LinearSVC_classifier")

filename6 = 'LinearSVC_classifier.sav'
pickle.dump(LinearSVC_classifier, open(dircetory + "Output\\Models\\" + filename6, 'wb'))


print("Training BernoulliNB_classifier")
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("Trained BernoulliNB_classifier")

filename7 = 'BernoulliNB_classifier.sav'
pickle.dump(BernoulliNB_classifier, open(dircetory + "Output\\Models\\" + filename7, 'wb'))

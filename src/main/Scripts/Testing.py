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
inpMessages = csv.reader(open(dircetory + 'Resources\\sms_spam_test.csv', 'r', encoding = "cp850"))
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

testing_set = nltk.classify.util.apply_features(extract_features, messages)

model_names = ["NaiveBayes_Classifier", "SVC_classifier", "LogisticRegression_classifier",
                "MNB_classifier", "SGDClassifier_classifier", "LinearSVC_classifier",
                "BernoulliNB_classifier"]

accuracy_list = []
for name in model_names:
    print("Now testing: " + name)
    classifier = pickle.load(open(dircetory + "Output\\Models\\" + name + ".sav", 'rb'))
    accuracy_percentage = (nltk.classify.accuracy(classifier, testing_set))*100
    accuracy_list.append(accuracy_percentage)

print(accuracy_list)
dict1 = {
    "Algorithm": model_names,
    "Accuracy Percentage": accuracy_list
}

df = pd.DataFrame(dict1, columns=["Algorithm", "Accuracy Percentage"])
df.to_csv(dircetory + "Output\\csv\\SpamAccuracy.csv", index=0)

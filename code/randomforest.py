from random import shuffle
from collections import Counter

class RandomForest:
    # treegen is a function taking a data set, features and returning a function
    # which takes a datapoint and returns a dict with key: class, value: probability
    def __init__(self, treegen, data, features, numtrees):
        fl = list(features)
        shuffle(fl)
        featuregroups = [fl[i::numtrees] for i in xrange(numtrees)]
        self.classifiers = [treegen(data, f) for f in featuregroups]

    def classprobabilities(self, point):
        predictions = [c(point) for c in self.classifiers]
        classprobs = Counter()
        for prediction in predictions:
            for c in prediction:
                classprobs[tuple(c)] += prediction[c]
        for c in classprobs:
            classprobs[c] /= len(predictions)
        return classprobs

    def classify(self, point):
        classprobs = self.classprobabilities(point)
        return max(classprobs, key=classprobs.get)

from math import log
from collections import Counter, defaultdict

class DecisionTree(object):
    def __init__(self, question, leaf):
        self.isleaf = leaf
        self.question = question
        self.children = {}

    def addChild(self, answer, tree):
        self.children[answer] = tree

    def printSplits(self, height = 0):
        if self.isleaf:
            return
        print self.question, height
        for child in self.children:
            self.children[child].printSplits(height + 1)

    def printTree(self, indent = 0):
        if self.isleaf:
            return
        print "\t" * indent + str(self.question)
        print "\t" * indent + "Paths:"
        for child in self.children:
            print "\t" * indent + str(child) + " -> " + str(self.children[child].question)
            self.children[child].printTree(indent + 1)

    def classprobabilities(self, observation):
        if self.isleaf:
            return self.question
        else:
            if observation['features'][self.question] in self.children:
                return self.children[observation['features'][self.question]].classprobabilities(observation)
            total = Counter()
            for child in self.children:
                probs = self.children[child].classprobabilities(observation)
                for c in probs:
                    total[c] += probs[c] / len(self.children)
            return dict(total)

    def classify(self, observation):
        classprobs = self.classprobabilities(observation)
        return max(classprobs, key=classprobs.get)


def entropy(distribution): 
    return -sum([0 if not p > 0 else p * log(p, 2) for p in distribution])
def normalize(distribution):
    count = sum(distribution.itervalues())
    if count == 0:
        return [value for value in distribution.itervalues()]
    return [float(value) / float(count) for value in distribution.itervalues()]

def id3(observations, features, maxdepth):
    def preprocess(observation):
        if type(observation['features']) == Counter:
            ppfeatures = observation['features']
        elif type(observation['features']) == dict:
            ppfeatures = Counter(observation['features'])
        else:
            ppfeatures = Counter({i: observation['features'][i] for i in xrange(len(observation['features']))})
        ppclasses = tuple(observation['classes'])
        return {'features': ppfeatures, 'classes': ppclasses}

    def id3helper(observations, depth, maxdepth):
        # actual id3 algorithm is here

        # if there is only one set of classes left, no need for further branching
        if len(list(set([observation["classes"] for observation in observations]))) == 1:
            return DecisionTree({observations[0]["classes"]: 1.0}, True)

        # if there are no discriminating features left or we are at max depth,
        # just choose majority class
        if len(observations[0]["features"]) == 0 or depth == maxdepth:
            counter = Counter([observation['classes'] for observation in observations])
            return DecisionTree({c: float(counter[c]) / len(observations) for c in counter}, True)

        # pick the feature which provides the most information gain
        entropies = {}
        for feature in features:
            # faster than counter
            distribution = defaultdict(lambda:defaultdict(lambda:0))
            for observation in observations:
                distribution[observation["features"][feature]][observation["classes"]] += 1
            entropies[feature] = sum([entropy(normalize(distribution[val])) * \
                                      sum(distribution[val].itervalues()) for val in distribution]) \
                                 / len(observations)
        best_gain = min(entropies, key=entropies.get)
        tree = DecisionTree(best_gain, False)
        bins = defaultdict(list)
        for observation in observations:
            bins[observation['features'][best_gain]].append(observation)
        for val in bins:
            tree.addChild(val, id3helper(bins[val], depth + 1, maxdepth))
        return tree

    return id3helper([preprocess(observation) for observation in observations], 0, maxdepth)

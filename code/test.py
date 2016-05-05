import sys
from parse import stateoftheunions
from collections import Counter, defaultdict
from nltk import tokenize
from id3 import DecisionTree, id3, normalize, entropy
from naivebayes import naivebayesclassify
from random import shuffle
from randomforest import RandomForest

start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.
unknown_token = "<UNK>"  # unknown word token.

def baselineclassifier(datapoint):
    return [1]

def stratifiedcrossvalidate(trainer, labeledsamples, k):
    def getfolds(l, n):
        return [l[i::n] for i in xrange(n)]

    # splits
    classbins = defaultdict(list)
    for sample in labeledsamples:
        classbins[sample['classes'][0]].append(sample)
    for b in classbins:
        shuffle(classbins[b])
    cuts = [[i for l in n for i in l] for n in zip(*[getfolds(classbins[b], k) for b in classbins])]

    # calculate accuracies
    accuracies = [None for i in xrange(k)]
    for cut in xrange(k):
        train = [i for j in xrange(len(cuts)) for i in cuts[j] if not j == cut]
        test = cuts[cut]
        classifier = trainer(train)
        accuracies[cut] = accuracy(classifier, test)

    return accuracies
    
    

def accuracy(classifier, labeledsamples):
    correct = 0
    for sample in labeledsamples:
        if tuple(classifier(sample)) == tuple(sample['classes']):
            correct += 1
    return float(correct) / len(labeledsamples)

# find the cut that (approximately) minimizes entropy
def decisionsplit(dataset, feature):
    totalclasscounts = Counter()
    for sample in dataset:
        totalclasscounts[sample['classes'][0]] += 1
    classcounts = Counter()
    featureorder = sorted(dataset, key=lambda s: s['features'][feature])
    splitentropies = {}
    # check entropy at all boundary points
    for i in xrange(len(featureorder) - 1):
        classcounts[featureorder[i]['classes'][0]] += 1
        if not featureorder[i]['classes'][0] == featureorder[i + 1]['classes'][0]:
            split = (featureorder[i]['features'][feature] + \
                     featureorder[i + 1]['features'][feature]) \
                    / 2.0
            leftent = entropy(normalize(classcounts))
            rightent = entropy(normalize({c: totalclasscounts[c] - classcounts[c] for c in totalclasscounts}))
            splitentropies[split] = (float(i + 1) / len(featureorder)) * leftent + \
                                    (float(len(featureorder) - i - 1) / len(featureorder)) * rightent
    return min(splitentropies, key=splitentropies.get)

# returns the set of things in things that occur at least minDups times
def AtLeastNDups(things, minDups):
    occurrences = {}
    for thing in things:
        if not thing in occurrences:
            occurrences[thing] = 1
        else:
            occurrences[thing] += 1
    return set([thing for thing in occurrences if occurrences[thing] >= minDups])

def preprocess(sample):
    # tokenize
    sample['speech'] = tokenize.sent_tokenize(sample['speech'])
    for i in xrange(len(sample['speech'])):
        sample['speech'][i] = tokenize.word_tokenize(sample['speech'][i])
        sample['speech'][i] = [word.lower() for word in sample['speech'][i]]
        sample['speech'][i] = [start_token] + sample['speech'][i] + [end_token]

def unkify(sample, wordset):
    for i in xrange(len(sample['speech'])):
        sample['speech'][i] = [word if word in wordset else unknown_token \
                               for word in sample['speech'][i]]

def computefeatures(dataset, wordset):
    # n-gram counts
    max_ngram_len = 4
    for sample in dataset:
        sample['ngrams'] = [Counter() for x in xrange(max_ngram_len)]
        for sent in sample['speech']:
            for i in xrange(len(sent)):
                for ngramlen in xrange(1, max_ngram_len + 1):
                    if ngramlen + i <= len(sent):
                        sample['ngrams'][ngramlen - 1][tuple(sent[i:i+ngramlen])] += 1


    # binary bag of words
    for sample in dataset:
        sample['bbow'] = {word: 0 for word in wordset}
        for sent in sample['speech']:
            for word in sent:
                sample['bbow'][word] = 1

    # tf-idf
    df = Counter()
    for sample in dataset:
        for sent in sample['speech']:
            for word in sent:
                df[word] += 1
    for sample in dataset:
        sample['tf-idf'] = Counter()
        for sent in sample['speech']:
            for word in sent:
                sample['tf-idf'][word] += 1.0 / df[word]

    # unify features for each sample into a single features vector
    # use a counter so features not present will read as 0 without having to store them
    for sample in dataset:
        sample['features'] = Counter()
        # ngrams
        for ngramlen in xrange(1, max_ngram_len + 1):
            for ngram in sample['ngrams'][ngramlen - 1]:
                sample['features'][('ngram', ngram)] = sample['ngrams'][ngramlen - 1][ngram]
        # bbow
        for word in sample['bbow']:
            sample['features'][('bbow', word)] = sample['bbow'][word]
        # tf-idf
        for word in sample['tf-idf']:
            sample['features'][('tf-idf', word)] = sample['tf-idf'][word]


def main(args):
    dataset = stateoftheunions()

    for sample in dataset:
        preprocess(sample)

    common_words = AtLeastNDups((word for sample in dataset \
                                      for sent in sample['speech'] \
                                      for word in sent), 4)

    for sample in dataset:
        unkify(sample, common_words)

    computefeatures(dataset, common_words.union(set(unknown_token)))

    for sample in dataset:
        sample['classes'] = [0] if sample['party'] == "Republican" else [1]

    featureset = set()
    for sample in dataset:
        featureset = featureset.union(set(sample['features'].keys()))

    #globalfeatureset = set((feature for feature in featureset if all((feature in sample['features'] for sample in dataset))))

    commonfeatureset = set((feature for feature in featureset if sum((1 if feature in sample['features'] else 0 for sample in dataset)) > 5))

    print "featureset len", len(featureset)
    print "commonfeaturest len", len(commonfeatureset)

    print "binarizing"

    splits = {feature: decisionsplit(dataset, feature) for feature in commonfeatureset}

    bfeatureset = set()

    for sample in dataset:
        sample['bfeatures'] = Counter()
        for feature in [f for f in sample['features'] if f in commonfeatureset]:
            bfeature = str(feature) + ' > ' + str(splits[feature])
            if not bfeature in bfeatureset:
                bfeatureset = bfeatureset.union(set([bfeature]))
            if sample['features'][feature] <= splits[feature]:
                sample['bfeatures'][bfeature] = 0
            else:
                sample['bfeatures'][bfeature] = 1

    print "filtering"

    featureentropy = {}
    for feature in bfeatureset:
        ent0 = entropy(normalize({feature: sample['classes'][0] for sample in dataset if sample['bfeatures'][feature] == 0}))
        ent1 = entropy(normalize({feature: sample['classes'][0] for sample in dataset if sample['bfeatures'][feature] == 1}))
        featureentropy[feature] = (len([s for s in dataset if s['bfeatures'][feature] == 0]) / len(dataset)) * ent0 + (len([s for s in dataset if s['bfeatures'][feature] == 1]) / len(dataset)) * ent1

    bestfeatures = sorted(featureentropy, key=featureentropy.get)[:1000]

    training = [{'features': Counter({f: sample['bfeatures'][f] for f in sample['bfeatures'] if f in bestfeatures}), 'classes': sample['classes']} for sample in dataset]

    baselinetrainer = lambda t: baselineclassifier

    baselinecv = stratifiedcrossvalidate(baselinetrainer, training, 5)

    print "cross-validated accuracy with baseline:", str(baselinecv)
    print "average accuracy:", str(sum(baselinecv) / 5.0)


    nbtrainer = lambda t: lambda s: naivebayesclassify(t, bestfeatures, s)

    nbcv = stratifiedcrossvalidate(nbtrainer, training, 5)

    print "cross-validated accuracy with naive bayes:", str(nbcv)
    print "average accuracy:", str(sum(nbcv) / 5.0)


    for maxdepth in [5, 10, 20, 40, 80]:

        id3trainer = lambda t: id3(t, bestfeatures, maxdepth).classify

        id3cv = stratifiedcrossvalidate(id3trainer, training, 5)

        print "cross-validated accuracy with id3 and max depth of", str(maxdepth), ":", str(id3cv)
        print "average accuracy:", str(sum(id3cv) / 5.0)



    for maxdepth in [5, 10, 20, 40, 80]:
        for numtrees in [64, 80, 96, 112, 128]:
            rftrainer = lambda t: RandomForest(lambda o,f: id3(o, f, maxdepth).classprobabilities, t, bestfeatures, numtrees).classify

            rfcv = stratifiedcrossvalidate(rftrainer, training, 5)

            print "cross-validated accuracy with random forest with", str(numtrees), "trees and", str(maxdepth), "max depth:", str(rfcv)
            print "average accuracy:", str(sum(rfcv) / 5.0)

    """
    use random forest since it has by far the highest accuracy
    random forest democrat/republican semantic differential:
    rf.classprobabilities(speech)[(1,)]
    where 0 is republican, 1 is democrat
    this is just the probability of democrat
    so when it is low, we are closer to republican (0)
    and when it is high, we are closer to democrat (1)

    """

    """
    for feature in bestfeatures:
        dem0 = sum([s['classes'][0] for s in training if s['features'][feature] == 0])
        dem1 = sum([s['classes'][0] for s in training if s['features'][feature] == 1])
        print feature, "TRUE", "Democrats:", str(dem1), "Republicans:", str(len(training) - dem1)
        print feature, "FALSE", "Democrats:", str(dem0), "Republicans:", str(len(training) - dem0)
        """

if __name__ == "__main__":
    main(sys.argv)

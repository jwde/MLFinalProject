from collections import Counter, defaultdict
from math import log

def naivebayesclassify(training, features, sample):
    classes = Counter()
    for s in training:
        classes[s['classes'][0]] += 1
    classprobs = {}
    featureprobs = defaultdict(lambda:defaultdict(Counter))
    for s in training:
        for feature in features:
            featureprobs[s['classes'][0]][feature][s['features'][feature]] += 1.0 / float(classes[s['classes'][0]])
            
    for c in classes:
        prior = log(float(classes[c])) - log(len(training))
        featureprob = reduce(lambda a, n: a + log(n) if n > 0 else float('-inf'), [featureprobs[c][f][sample['features'][f]] for f in features])
        classprobs[c] = prior + featureprob
    return [max(classprobs, key=classprobs.get)]

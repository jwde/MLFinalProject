from itertools import groupby

def stateoftheunions(path = '../stateoftheunion/dataset'):
    f = open(path)
    lines = [line.rstrip() for line in f.readlines()]
    f.close()
    delim = "%%%%STOP%%%%"
    speechgroups = [list(line) for d, line in groupby(lines, lambda l: l == delim) if not d]
    speeches = []
    for speechgroup in speechgroups:
        speech = {}
        for i in xrange(0, len(speechgroup), 2):
            key = 'president' if speechgroup[i] == '%%%%PRESIDENT%%%%' else \
                  'party' if speechgroup[i] == '%%%%PARTY%%%%' else \
                  'date' if speechgroup[i] == '%%%%DATE%%%%' else \
                  'speech'
            speech[key] = speechgroup[i + 1]
        speeches.append(speech)
    return speeches

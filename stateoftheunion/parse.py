import re
import os

parties = {'Franklin D. Roosevelt': 'Democrat',
           'Harry S. Truman': 'Democrat',
           'Dwight D. Eisenhower': 'Republican',
           'John F. Kennedy': 'Democrat',
           'Lyndon B. Johnson': 'Democrat',
           'Richard Nixon': 'Republican',
           'Gerald R. Ford': 'Republican',
           'Jimmy Carter': 'Democrat',
           'Ronald Reagan': 'Republican',
           'George H.W. Bush': 'Republican',
           'William J. Clinton': 'Democrat',
           'George W. Bush': 'Republican',
           'Barack Obama': 'Democrat'}

def parsefile(f):
    contents = " ".join([l.rstrip() for l in f])

    president = re.search(r'<h2>(.*)</h2>', contents).group(1).rstrip()
    date = re.search(r'<h3>(.*)</h3>', contents).group(1).rstrip()
    unfilteredspeech = re.search(r'</h3>(.*)</p>', contents, re.M).group(1)

    tags = re.compile(r'<.*?>')
    tabs = re.compile(r'\t')

    filteredspeech = tabs.sub('', tags.sub('', unfilteredspeech))

    return {'president': president, 'date': date, 'speech': filteredspeech}


files = []
for filename in os.listdir("raw"):
    path = r"raw/" + filename
    f = open(path)
    files.append(f.readlines())
    f.close()

parsed = [parsefile(f) for f in files]
for (i, p) in enumerate(parsed):
    print r"%%%%PRESIDENT%%%%"
    print p['president']
    print r"%%%%PARTY%%%%"
    print parties[p['president']]
    print r"%%%%DATE%%%%"
    print p['date']
    print r"%%%%SPEECH%%%%"
    print p['speech']
    if not i == len(parsed):
        print r"%%%%STOP%%%%"

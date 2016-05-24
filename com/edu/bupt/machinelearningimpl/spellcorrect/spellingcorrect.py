import re, collections

class SpellingCorrect:
    def __init__(self):
        pass

    def __init__(self, path):
        self.path = path
        self.NWORDS = self.training(self.word(file(path).read()))
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def word(self, text):
        return re.findall('[a-z]+', text.lower())

    def training(self, feature):
        model = collections.defaultdict(lambda :1)
        for f in feature:
            model[f] += 1
        return model

    def edit_1(self, word):
        n = len(word)
        return set(
            [word[0:i]+word[i+1:] for i in range(n)]+
            [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)]+
            [word[0:i]+c+word[i+1:] for i in range(n) for c in self.alphabet]+
            [word[0:i]+c+word[i:] for i in range(n) for c in self.alphabet]
        )

    def edit_2(self, word):
        return set(e2 for e1 in self.edit_1(word) for e2 in self.edit_1(e1) if e2 in self.NWORDS)

    def know(self, words):
        return set(w for w in words if w in self.NWORDS)


    def correct(self, word):
        # p(w|c)
        candidates = self.know([word]) or self.know(self.edit_1(word)) or self.edit_2(word) or [word]
        # p(c)
        return max(candidates, key=lambda w: self.NWORDS[w])


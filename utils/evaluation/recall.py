
class Recall:
    def __init__(self, n):
        self.n = n
        self.hits = 0
        self.stream = 0
        
    def update(self, rank):
        if rank < self.n:
            self.hits += 1
        self.stream += 1
        
    def score(self):
        return float(self.hits/self.stream)


class MultipleRecall:
    def __init__(self, K = [1, 5, 10, 20]):
        self.hits = {}
        self.size = 0
        self.K = K
        self.initialize()

    def inilialize():
        for value in self.K:
            hits[value] = 0
        
    def update(self, rank):
        
        for k in self.hits.keys():
            if rank < k:
                self.hits[k] += 1

        self.size += 1
        
    def score(self):
        scores = []
        for k in self.hits.keys():
            score = round((self.hits[k]/self.size),3)
            scores.append(score)
        
        return tuple(scores)


class WindowedRecall:
    def __init__(self, n, win_size):
        self.n = n
        self.win_size = win_size 
        self.hits = 0
        self.stream = 0 
        self.r_mean = 0.0
        self.recall_list = []
        self.ranks = []

    def update(self, rank):
        if rank < self.n:
            self.hits = 1
        else:
            self.hits = 0
            
        self.ranks.append(self.hits)
        
        if(len(self.ranks) > self.win_size):
            self.ranks.pop(0)

    def score():
        sum(self.ranks)/len(self.ranks)
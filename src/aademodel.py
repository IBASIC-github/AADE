#-*- encoding: utf-8 -*-
'''
Created on 2015-08-10

@author: Baijie
'''
import sys
sys.path.append("..")
import random, math
import numpy as np
from gensim import matutils


class Model(object):
    def __init__(self, w2v):
        self.w2v = w2v
    
    def build_word_base(self):
        
        th_stopword = 200
        word_count = np.array([0]*len(self.w2v.vocab))
        for (_,voc) in self.w2v.vocab.items():
            word_count[voc.index] = voc.count
        s = word_count[th_stopword]
        word_count = list(i*1.0/s for i in word_count)
        for i in range(th_stopword):
            word_count[i] = 1
        s = sum(word_count)
        self.word_base = list(i/s for i in word_count)
    
    def activate_base(self, stimulants=[], topn = 10, beta = 0):

        self.w2v.init_sims()

        if isinstance(stimulants, basestring):
            positive = [stimulants]

        positive = [(word, 1.0) for word in stimulants]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive:
            if word in self.w2v.vocab:
                mean.append(weight * self.w2v.syn0norm[self.w2v.vocab[word].index])
                all_words.add(self.w2v.vocab[word].index)

        if not mean:
            raise ValueError("cannot compute similarity with no input")
            return []
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        
        if beta == 0:
            dists = np.dot(self.w2v.syn0norm, mean)
        else:
            dists = (1.0-beta)*np.dot(self.w2v.syn0norm, mean) + np.dot(beta,self.word_base)
        
        if not topn:
            best = matutils.argsort(dists, reverse=True)
            result = [self.w2v.index2word[sim] for sim in best if sim not in all_words and float(dists[sim]) > 0]
            return result
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        result = [self.w2v.index2word[sim] for sim in best if sim not in all_words and float(dists[sim]) > 0]
        return result[:topn]


class GlobalEqualBase(Model):
      
    def __init__(self, w2v):
        super(GlobalEqualBase, self).__init__(w2v)
        self.modelname = 'Global equal base'
        
    def activation(self, dataset, alpha = 1.0, beta = 0):
        
        if alpha <= 0:
            return [[]*len(dataset)]
        
        actv_data = []
        for i in range(0, len(dataset)):
            doc = dataset[i]
            actv_doc = []
            if len(doc) != 0:
                numofwords = int(math.ceil(alpha*len(doc)))
                if numofwords >= 1:
                    actv_doc = self.activate_base(doc, topn = numofwords, beta = 0)
            actv_data.append(actv_doc)
        return actv_data


class GlobalActv(Model):
      
    def __init__(self, w2v):
        
        super(GlobalActv, self).__init__(w2v)
        self.modelname = 'Global Actv'
        self.build_word_base()
        
    def activation(self, dataset, alpha = 1.0, beta = 0.2):
        
        if alpha <= 0:
            return [[]*len(dataset)]
        
        actv_data = []
        for i in range(0, len(dataset)):
            doc = dataset[i]
            actv_doc = []
            if len(doc) != 0:
                numofwords = int(math.ceil(alpha*len(doc)))
                if numofwords >= 1: 
                    actv_doc = self.activate_base(doc, topn = numofwords, beta = beta)
            actv_data.append(actv_doc)
        return actv_data


class LocalActv(Model):
      
    def __init__(self, w2v):
        
        super(LocalActv, self).__init__(w2v)
        self.modelname = 'Local Actv'
        self.build_word_base()
        
    def activation(self, dataset, alpha = 1.0, beta = 0.2):
        if alpha <= 0:
            return [[]*len(dataset)]
        
        actv_data = []
        for i in range(0, len(dataset)):
            doc = dataset[i]
            actv_doc = []
            if len(doc) != 0:
                numofwords = int(math.ceil(alpha*len(doc)))
                if numofwords >= 1:
                    if len(doc) <= 3:
                        actv_doc = self.activate_base(doc, topn = numofwords, beta = beta)
                    else:
                        num_activated = 0
                        while num_activated < numofwords:
                            num_stimulant = random.randint(1,len(doc))
                            num_targets = int(math.ceil(num_stimulant*alpha))
                            
                            if num_targets < 1:
                                num_targets = 1
                            num_activated += num_targets
                            if num_activated > numofwords:
                                num_targets -= num_activated - numofwords
                                
                            index = random.randint(0, len(doc)-num_stimulant)
                            stimulant = doc[index:index+num_stimulant]
                            actv_doc += self.activate_base(stimulant, topn = numofwords, beta = beta)
            actv_data.append(actv_doc)
        return actv_data
    

class Similarity(Model):
      
    def __init__(self, w2v):
        
        super(Similarity, self).__init__(w2v)
        self.modelname = 'Similarity'
        
    def activation(self, dataset, alpha = 1.0, beta = 0):
        if alpha <= 0:
            return [[]*len(dataset)]
        
        actv_data = []
        top = int(math.ceil(alpha))
        bottom = int(math.floor(alpha))
        for i in range(0, len(dataset)):
            doc = dataset[i]
            actv_doc = []
            if len(doc) != 0:
                for word in doc:
                    rand = random.uniform(bottom, top)
                    if alpha <= rand:
                        numofwords = bottom
                    else:
                        numofwords = top
                    if numofwords <= 0:
                        continue
                    actv_doc += self.activate_base([word], topn = numofwords, beta = 0)
            actv_data.append(actv_doc)
        return actv_data


class Random(Model):
      
    def __init__(self, w2v):
        
        super(Random, self).__init__(w2v)
        self.modelname = 'Random'
        
    def activation(self, dataset, alpha = 1.0, beta = 0.2):
        if alpha <= 0:
            return [[]*len(dataset)]
        actv_data = []
        for doc in dataset:
            numofwords = int(math.ceil(alpha*len(doc)))
            if numofwords > len(self.w2v.index2word):
                numofwords = len(self.w2v.index2word)
            actv_doc = random.sample(self.w2v.index2word, numofwords)
            actv_data.append(actv_doc)
        return actv_data
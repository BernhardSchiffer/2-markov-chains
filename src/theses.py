#!/usr/bin/env python3

# %%
# we will be using nltk.lm and numpy
from cgitb import lookup
from pyexpat import model
from matplotlib.pyplot import text
from matplotlib.style import context
import numpy as np
from nltk import everygrams
from nltk.lm import MLE
from nltk.probability import FreqDist
from nltk.lm.preprocessing import padded_everygram_pipeline
import random


# 0. Before you get started, make sure to download the `theses.txt` data set.

def get_data() -> list :
    f = open('../res/theses.txt', 'r')
    lines = f.readlines()
    f.close()

    for idx, line in enumerate(lines):
        lines[idx] = line.replace("\n", "")    
    return lines

# 1. Spend some time on pre-processing. How would you handle hyphenated words
#    and abbreviations/acronyms?

def tokenize(data: str) -> list:
    return data.split()

# 2. Train n-gram models with n = [1, ..., 5]. What about <s> and </s>?

def train_model(data, n: int):
    model = MLE(n)
    train, vocab = padded_everygram_pipeline(n, data)
    model.fit(train, vocab)
    return model

def get_everygrams(data: list, min_n: int, max_n: int):
    grams = []
    for idx, sentence in enumerate(data):
        gram = everygrams(
            sentence,
            min_len=min_n, 
            max_len=max_n, 
            pad_left=True, 
            pad_right=True,
            left_pad_symbol="<s>",
            right_pad_symbol="</s>")
        grams.append(gram)

    return grams

data = get_data()
tokenized_titles = [tokenize(title) for title in data]

trained_models = {}
for n in range(1, 6):
    model = train_model(tokenized_titles, n)
    print(f'{n}-gram model finished training')
    trained_models[n] = model

# %%

# 3. Write a generator that provides thesis titles of desired length. Please
#    do not use the available `lm.generate` method but write your own.
#    nb: If you fix the seed in numpy.random.choice, you get reproducible 
#        results.
# 3.1 How can you incorporate seed words?
# 3.2 How do you handle </s> tokens (w.r.t. the desired length?)

class MaxSizeList(object):

    def __init__(self, max_length):
        self.max_length = max_length
        self.list = []

    def push(self, st):
        self.list.append(st)
        if len(self.list) > self.max_length:
            self.list.pop(0)

    def get_list(self):
        return self.list

def generate_lookup(ngrams: list):
    fdist = FreqDist()

    for entry in ngrams:
        fdist.update(list(entry))

    lookup = {}
    for ngram in fdist:        
        key =  ngram[:-1]
        word = ngram[-1]
        count = fdist[ngram]

        if key not in lookup:
            lookup[key] = {}
        
        lookup[key][word] = count

    return lookup

def generate_thesis(ngrams: list, n: int, text_seed: str = '<s>', length: int = 10):
    
    
    def get_next(context: tuple):
        values = lookup[context]

        choices = []
        counts =  []

        for choice in values.keys(): 
            choices.append(choice)
            counts.append(values[choice])

        word = random.choices(choices, weights=counts, k=1)
        return word[0]

    lookup = generate_lookup(ngrams)
    context = MaxSizeList(n-1)
    context.push(text_seed)

    title = []
    counter = 0
    while counter <= length:
        w = get_next(tuple(context.get_list()))
        if w == '<s>':
            continue

        if w == '</s>':
            break

        title.append(w)        
        context.push(w)

    return title
# %%

# 3.3 If you didn't just copy what nltk's lm.generate does: compare the
#     outputs

num_words = 10

def remove_items(given_list, item):
    # using list comprehension to perform the task
    res = [i for i in given_list if i != item]
    return res


for n in range(2,6):
    print(f'--------- {n} ---------')
    generated_tokens = trained_models[n].generate(num_words, text_seed=['<s>'], random_seed=42)
    generated_tokens = remove_items(generated_tokens, '<s>')
    generated_tokens = remove_items(generated_tokens, '</s>')
    
    print(' '.join(generated_tokens))

    ngrams = get_everygrams(tokenized_titles, 1, n)
    thesis = generate_thesis(ngrams=ngrams, text_seed='<s>', length=num_words, n=n)
    print(' '.join(thesis))

# %%

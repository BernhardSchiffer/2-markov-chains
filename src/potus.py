#!/usr/bin/env python3
# %%
# we will be using nltk.lm
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# %%
# 0. Before you get started, make sure to download the Obama and Trump twitter
#    archives.

# Since the nltk.lm modules will work on tokenized data, implement a 
# tokenization method that strips unnecessary tokens but retains special
# words such as mentions (@...) and hashtags (#...).

def tokenize(sentence: str) -> list:
    sentence = sentence.lower()

    # replace urls
    sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<url>', sentence)
    
    # replace special chars with spaces
    for ch in [',','.','!','?','-','-','_','\s',':','"']:
        if ch in sentence:
            sentence = sentence.replace(ch,' ')

    # split string on spaces
    tokens = sentence.split()

    return tokens

#print(tokenize('Hello World.https://funzt.cool/hello/world?q=12332 what the hell?This is great\s.hello . world'))

# 1. Prepare all the tweets, partition into training and test sets; select
#    about 100 tweets each, which we will be testing on later.
#    nb: As with any ML task, training and test must not overlap

def get_tokenized_tweets(df: pd.DataFrame, key: str):
    tweets = df[key].reset_index(drop=True).to_frame()
    tweets = tweets[key].apply(lambda x: tokenize(x)).to_frame()
    return tweets

def split_train_test(data, test_size, random_state=42):
    randomized = data.sample(frac=1, random_state = random_state).reset_index(drop=True)
    length = data.size
    #test_size = math.floor(test_size * length)
    train = randomized.head(length-test_size).reset_index(drop=True)
    test = randomized.tail(test_size).reset_index(drop=True)
    return (train, test)

trump_tweets = pd.read_json("../res/tweets_01-08-2021.json")
trump_tweets = trump_tweets[trump_tweets.isRetweet == 'f']
trump_tweets = get_tokenized_tweets(trump_tweets, "text")
display(trump_tweets.head(5))
trump_train, trump_test = split_train_test(data=trump_tweets, test_size=100)

obama_tweets = pd.read_csv("../res/Tweets-BarackObama.csv")
obama_tweets = get_tokenized_tweets(obama_tweets, 'Tweet-text')
display(obama_tweets.head(5))
obama_train, obama_test = split_train_test(data=obama_tweets, test_size=100)

biden_tweets = pd.read_csv("../res/JoeBidenTweets.csv")
biden_tweets = get_tokenized_tweets(biden_tweets, 'tweet')
display(biden_tweets.head(5))
biden_train, biden_test = split_train_test(data=biden_tweets, test_size=100)

# %%
# 2. Train n-gram models with n = [1, ..., 5] for both Obama, Trump and Biden.
# 2.1 Also train a joint model, that will serve as background model

def train_model(data, n: int):
    model = MLE(n)
    train, vocab = padded_everygram_pipeline(n, data)
    model.fit(train, vocab)
    return model

trained_ngrams = {}
for n in range(1, 6):
    print(n)
    trump_model = train_model(trump_train['text'].tolist(), n)
    print(f'trump {n} finished')
    obama_model  = train_model(obama_train['Tweet-text'].tolist(), n)
    print(f'obama {n} finished')
    biden_model  = train_model(biden_train['tweet'].tolist(), n)
    print(f'biden {n} finished')
    combined_train = []
    combined_train.extend(trump_train['text'].tolist())
    combined_train.extend(obama_train['Tweet-text'].tolist())
    combined_train.extend(biden_train['tweet'].tolist())
    combined_model = train_model(combined_train, n)
    print(f'combined {n} finished')

    trained_ngrams[n] = {
        "trump": trump_model, 
        "obama": obama_model, 
        "biden": biden_model, 
        "combined": combined_model
    }
# %%

print(trained_ngrams[3]['trump'].generate(20))
#print(list(trained_ngrams[3]['trump'].vocab)[:50])

# %%
# 3. Use the log-ratio method to classify the tweets. Trump should be easy to
#    spot; but what about Biden vs. Trump?
# 3.1 Analyze: At what context length (n) does the system perform best?

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

def get_score(text: list, model: MLE):
    prev_words = MaxSizeList(model.order - 1)
    score = 1
    for word in text:
        # map unknown words to <UNK> token
        word = model.vocab.lookup(word)

        # calc score
        tmp_score = model.logscore(word, prev_words.get_list())
        if(tmp_score != math.inf and tmp_score != -math.inf):
            score = score * model.logscore(word, prev_words.get_list())
            
        #print(f'{word} \t| {prev_words.get_list()} \t= {model.logscore(word, prev_words.get_list())}')
        prev_words.push(word)

    return abs(score)

#print(trump_test['text'][11])
#print(get_score(trump_test['text'][11], trained_ngrams[3]['obama']))


def predict(text: list, models, keys: list, ngram: int):

    scores = []
    for key in keys:
        score = (get_score(text, models[ngram][key]), key)
        scores.append(score)

    #print(scores)
    x = scores[0][0] / scores[1][0] if scores[1][0] else 0
    s = math.log(x) if x !=0 else -math.inf
    if s > 0:
        return scores[0][1]
    else:
        return scores[1][1]
    #maximum = max(scores, key=lambda x: x[0])
    #return maximum[1]

#print(trump_test['text'][13])
#print(predict(trump_test['text'][13], trained_ngrams, ['trump', 'obama'], 3))

def plot_ConfusionMatrix(df):
    encoder = LabelEncoder()
    encoder.fit(df['actual'])
    act = encoder.transform(df['actual'])
    pred = encoder.transform(df['predicted'])

    cm = confusion_matrix(act, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=encoder.classes_)
    disp.plot()
    plt.show()

print('trump vs. obama')
for n in range(1,6):
    print(n)
    df = pd.DataFrame(columns=['actual', 'predicted'])
    for tweet in trump_test['text']:
        df.loc[len(df.index)] = ['trump', predict(tweet, trained_ngrams, ['trump', 'obama'], n)]
    for tweet in obama_test['Tweet-text']:
        df.loc[len(df.index)] = ['obama', predict(tweet, trained_ngrams, ['trump', 'obama'], n)]

    plot_ConfusionMatrix(df)

print('trump vs. biden')
for n in range(1,6):
    print(n)
    df = pd.DataFrame(columns=['actual', 'predicted'])
    for tweet in trump_test['text']:
        df.loc[len(df.index)] = ['trump', predict(tweet, trained_ngrams, ['trump', 'biden'], n)]
    for tweet in biden_test['tweet']:
        df.loc[len(df.index)] = ['biden', predict(tweet, trained_ngrams, ['trump', 'biden'], n)]

    plot_ConfusionMatrix(df)

print('biden vs. obama')
for n in range(1,6):
    print(n)
    df = pd.DataFrame(columns=['actual', 'predicted'])
    for tweet in biden_test['tweet']:
        df.loc[len(df.index)] = ['biden', predict(tweet, trained_ngrams, ['biden', 'obama'], n)]
    for tweet in obama_test['Tweet-text']:
        df.loc[len(df.index)] = ['obama', predict(tweet, trained_ngrams, ['biden', 'obama'], n)]

    plot_ConfusionMatrix(df)

# %%

# 4. Compute (and plot) the perplexities for each of the test tweets and 
#    models. Is picking the Model with minimum perplexity a better classifier
#    than in 3.?

# %%

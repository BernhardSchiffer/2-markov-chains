#!/usr/bin/env python3

# we will be using nltk.lm
import nltk


# 0. Before you get started, make sure to download the Obama and Trump twitter
#    archives.

# Since the nltk.lm modules will work on tokenized data, implement a 
# tokenization method that strips unnecessary tokens but retains special
# words such as mentions (@...) and hashtags (#...).

# 1. Prepare all the tweets, partition into training and test sets; select
#    about 100 tweets each, which we will be testing on later.
#    nb: As with any ML task, training and test must not overlap


# 2. Train n-gram models with n = [1, ..., 5] for both Obama, Trump and Biden.
# 2.1 Also train a joint model, that will serve as background model


# 3. Use the log-ratio method to classify the tweets. Trump should be easy to
#    spot; but what about Biden vs. Trump?
# 3.1 Analyze: At what context length (n) does the system perform best?

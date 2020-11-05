# Chatbot with Deep NLP

# importing libraries
import numpy as np
import tensorflow as tf
import re
import time
import itertools
##### Data preprocessing #####

# read in the data
dat_lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
dat_convs = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# set up the dictionay
id_to_line = {}
for line in dat_lines:
    newline = line.split(' +++$+++ ')
    if len(newline) == 5:
        id_to_line[newline[0]] = newline[-1]

## functions for iterating certain amount keys and values
def readDict(dictionary,N):
    out = dict(itertools.islice(dictionary.items(), N))    
    print("Dictionary limited by K is : " + str(out)) 

readDict(id_to_line,5)

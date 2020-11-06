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

# functions for iterating certain amount keys and values
def readDict(dictionary,N):
    out = dict(itertools.islice(dictionary.items(), N))    
    print("Dictionary limited by K is : " + str(out)) 


# setting up a list conversation
conversation_id = []
for conv in dat_convs:
    newconv = conv.split(' +++$+++ ')[-1]
    tempconv = newconv[1:-1].replace("'", "").replace(" ", "")
    conversation_id.append(tempconv.split(','))
conversation_id.remove(['']) # cleaning out empty rows

'''
backup information for the conversation id, 
the first number represents the question, 
the second for answer
'''

# seperating questions and answers from the conversation
questions = []
answers = []
for conversation in conversation_id:
    for i in range(len(conversation) - 1):
        questions.append(id_to_line[conversation[i]])
        answers.append(id_to_line[conversation[i + 1]])
# print(questions[26])
# print(answers[26])

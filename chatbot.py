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
        id_to_line[newline[0]] = newline[4]

# function for iterating certain amount keys and values
def readDict(dictionary,N):
    out = dict(itertools.islice(dictionary.items(), N))    
    print("Dictionary limited by K is : " + str(out)) 


# setting up a list conversation
conversation_id = []
for conv in dat_convs[:-1]:
    tempconv = conv.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversation_id.append(tempconv.split(','))
# conversation_id.remove(['']) # cleaning out empty rows

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

# function for cleaning the text
def cleanText(text):
    text = text.lower()
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# cleaning up the question
cleanQuestion = []
for question in questions:
    tempQuestion = cleanText(question)
    cleanQuestion.append(tempQuestion)

# cleaning up the answers
cleanAnswer = []
for answer in answers:
    tempAnswer = cleanText(answer)
    cleanAnswer.append(tempAnswer)

# counting word frequency
wordCount = {}
for question in cleanQuestion:
    for word in question.split():
        if word not in wordCount:
            wordCount[word] = 1
        else:
            wordCount[word] += 1

for answer in cleanAnswer:
    for word in answer.split():
        if word not in wordCount:
            wordCount[word] = 1
        else:
            wordCount[word] += 1

# mapping words with unique integer
threshold_question = 20
wordNum = 0
questionDic = {}
for word, count in wordCount.items():
    if count >= threshold_question:
        questionDic[word] = wordNum
        wordNum = wordNum + 1


threshold_answer = 20
answerDic = {}
wordNum = 0
for word, count in wordCount.items():
    if count >= threshold_answer:
        answerDic[word] = wordNum
        wordNum = wordNum + 1


# Extracting specifix keys from dictionary 
def filter_key(dictionary, filterKey):
    res = [dictionary[key] for key in filterKey] 
    print("Filtered value list is : " +  str(res)) 

# Adding last tokens to dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

'''
<PAD>:  GPU (or CPU at worst) processes your training data in batches and all the sequences in your batch should have the same length
<EOS>: end of string, Decoder (ending signal)
<OUT>: filtered out token
<SOS>: start of string, Decoder (initializing signal)
'''

for token in tokens:
    questionDic[token] = len(questionDic) + 1

for token in tokens:
    answerDic[token] = len(answerDic) + 1


# Invers the answer dictionary
answerDicInv = {w_i: w for w, w_i in answerDic.items()}

# Adding end of string token to the end of sentence inside clean answer list
for sentence in range(len(cleanAnswer)):
    cleanAnswer[sentence] += ' <EOS>'

# Translating clean questions to integers and replace filtered out words
questionInt = []
for question in cleanQuestion:
    Int = []
    for word in question.split():
        if word not in questionDic:
            Int.append(questionDic['<OUT>'])
        else:
            Int.append(questionDic[word])
    questionInt.append(Int)

answerInt = []
for answer in cleanAnswer:
    Int = []
    for word in answer.split():
        if word not in answerDic:
            Int.append(answerDic['<OUT>'])
        else:
            Int.append(answerDic[word])
    answerInt.append(Int)


# Sorting questions and answers by length of questions
sortedQuestion = []
sortedAnswer = []
for num in range(1, 26):
    for i in enumerate(questionInt):
        if len(i[1]) == num:
            sortedQuestion.append(questionInt[i[0]])
            sortedAnswer.append(answerInt[i[0]])


##### building sequence to sequence model #####

# building functions for place holders
def model_placeHolder():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'inputs')
    targets = tf.placeholder(tf.int32, [None, None], name = 'targets')
    learningRates = tf.placeholder(tf.int32, name = 'learningRates')
    keepProb = tf.placeholder(tf.int32, name = 'keepProb')
    return inputs, targets, learningRates, keepProb

# target preprocessing function
def preprocessing(target, wordInt, batchSize):
    begin = tf.fill([batchSize, 1], wordInt['<SOS>'])
    end = tf.strided_slice(target, [0,0], [batchSize - 1], [1,1])
    preprocessedTarget = tf.concat([begin, end], 1)
    return preprocessedTarget

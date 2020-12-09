# Chatbot with Deep NLP

# importing libraries
import numpy as np
import tensorflow as tf
import re
import time
import itertools

##### Data preprocessing #####

# read in the data
dat_lines = open('./data/movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
dat_convs = open('./data/movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

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
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    keepProb = tf.placeholder(tf.float32, name = 'keepProb')
    return inputs, targets, learning_rate, keepProb

# target preprocessing function
def preprocessing_targets(targets, word2Int, batchSize):
    begin = tf.fill([batchSize, 1], word2Int['<SOS>'])
    end = tf.strided_slice(targets, [0,0], [batchSize, - 1], [1,1])
    preprocessedTarget = tf.concat([begin, end], 1)
    return preprocessedTarget


# Encoder Recurrent Neural Network Layer
def encoderRNN(rnnInputs, rnnSize, layers, keepProb, seqLength):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnnSize)
    lstmDropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keepProb)
    encoderCell = tf.contrib.rnn.MultiRNNCell([lstmDropout] * layers)
    encoderOutput, encoderState = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoderCell,
                                                    cell_bw = encoderCell,
                                                    sequence_length = seqLength,
                                                    inputs = rnnInputs,
                                                    dtype = tf.float32)
    return encoderState


# Decoder for the training set
def decode_trainingSet(encoderState, decoderCell, decoder_embedded_input, sequence_length, decodingScope, output_function, keepProb, batchSize):
    attention_states = tf.zeros([batchSize, 1, decoderCell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoderCell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoderState[0],
                                                                            attention_keys,
                                                                            attention_values,
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoderCell,
                                                                                                            training_decoder_function,
                                                                                                            decoder_embedded_input,
                                                                                                            sequence_length,
                                                                                                            scope = decodingScope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keepProb)
    return output_function(decoder_output_dropout)

# Decoder for the testing set
def decode_testingSet(encoderState, decoderCell, decoder_embedded_matrix, sos_id, eos_id, maxLength, num_words, decodingScope, output_function, keepProb, batchSize):
    attention_states = tf.zeros([batchSize, 1, decoderCell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoderCell.output_size)
    testing_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                            encoderState[0],
                                                                            attention_keys,
                                                                            attention_values,
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            decoder_embedded_matrix,
                                                                            sos_id,
                                                                            eos_id,
                                                                            maxLength,
                                                                            num_words,
                                                                            name = "attn_dec_inf")
    testPredictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoderCell,
                                                                                                            testing_decoder_function,
                                                                                                            scope = decodingScope)
    return testPredictions

# Decoder Reccurent Neural Network
def decoderRNN(decoder_embedded_input, decoder_embedded_matrix, encoderState, num_words, seqLength, rnnSize, numLayers, wordInt, keepProb, batchSize):
    with tf.variable_scope("decoding") as decodingScope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnnSize)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keepProb)
        decoderCell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * numLayers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                    num_words,
                                                                    None,
                                                                    scope = decodingScope,
                                                                    weights_initializer = weights,
                                                                    biases_initializer = biases
                                                                    )
        training_predictions = decode_trainingSet(encoderState,
                                                decoderCell,
                                                decoder_embedded_input,
                                                seqLength,
                                                decodingScope,
                                                output_function,
                                                keepProb,
                                                batchSize)

        decodingScope.reuse_variables()

        testing_predictions = decode_testingSet(encoderState,
                                                decoderCell,
                                                decoder_embedded_matrix,
                                                wordInt['<SOS>'],
                                                wordInt['<EOS>'],
                                                seqLength - 1,
                                                num_words,
                                                decodingScope,
                                                output_function,
                                                keepProb,
                                                batchSize)
    
    return training_predictions, testing_predictions


# Building sequence to sequence model
def seq2seq_model(inputs, targets, keepProb, batchSize, seqLength, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnnSize, numLayers, questionDic):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                            answers_num_words + 1,
                                                            encoder_embedding_size,
                                                            initializer = tf.random_uniform_initializer(0,1)) 
    encoder_state = encoderRNN(encoder_embedded_input, rnnSize, numLayers, keepProb, seqLength)
    preproccessed_targets = preprocessing_targets(targets, questionDic, batchSize)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0 ,1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preproccessed_targets)
    training_predictions, testing_predictions = decoderRNN(decoder_embedded_input,
                                                        decoder_embeddings_matrix,
                                                        encoder_state,
                                                        questions_num_words,
                                                        seqLength,
                                                        rnnSize,
                                                        numLayers,
                                                        questionDic,
                                                        keepProb,
                                                        batchSize
                                                        )
    return training_predictions, testing_predictions 


##### Training the sequence to sequence model #####

# Setting up the hyperparameters
epochs = 100
batchSize = 64
rnnSize = 512
numLayers = 3
encoder_embedding_size = 512
decoder_embedding_size = 512
learning_rate = 0.01
learningRates_decay = 0.9
min_learningRates = 0.0001
keepProb = 0.5

# Define the tensorflow session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Load in model inputs
inputs, targets, learning_rate, keepProb = model_placeHolder()

# Setting up the sequence length
seqLength = tf.placeholder_with_default(25, None, name = 'seqLength')

# Getting the input shape
input_shape = tf.shape(inputs)

# Getting training, testing prediction
training_predictions, testing_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                        targets,
                                                        keepProb,
                                                        batchSize,
                                                        seqLength,
                                                        len(answerDic),
                                                        len(questionDic),
                                                        encoder_embedding_size,
                                                        decoder_embedding_size,
                                                        rnnSize,
                                                        numLayers,
                                                        questionDic)

# Setting the loss error, optimizer, and gradient clipping
with tf.name_scope('optimization'):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                    targets,
                                                    tf.ones([input_shape[0], seqLength]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0, 5.0 ), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor != None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding sequences with <PAD> token
# Question: 'Who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD>, <PAD>, <PAD> 
# Answer: <SOS>, 'I', 'am', 'a', 'student', 'from', 'NCTU', '.', <EOS>, <PAD>
# Matching question and answer with the same length 
def apply_padding(batch_of_seq, word2int):
    max_seqLength = max([len(seq) for seq in batch_of_seq])
    return [seq + [word2int['<PAD>']] * (max_seqLength - len(seq)) for seq in batch_of_seq]

# Splitting data to batches of questions and answers
def split_to_batches(questions, answers, batchSize):
    for batch_index in range(0, len(questions) // batchSize):
        start_index = batch_index * batchSize
        questions_in_batch = questions[start_index : start_index + batchSize]
        answers_in_batch = answers[start_index : start_index + batchSize]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionDic))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerDic))
        yield padded_questions_in_batch, padded_answers_in_batch

# Splitting questions and amswers to training and validating 
training_validation_split = int(len(sortedQuestion) * 0.15)
training_questions = sortedQuestion[training_validation_split: ]
training_answers = sortedAnswer[training_validation_split: ]
validated_questions = sortedQuestion[ :training_validation_split]
validated_answers = sortedAnswer[ :training_validation_split]


# Training
batchIndex_check_trainingLoss = 100
batchIndex_check_validationLoss = ((len(training_questions)) // batchSize // 2) - 1
totalTraining_lossError = 0
listValidation_lossError = []
earlyStopping_check = 0
earlyStopping_stop = 1000
chechpoint = "chatbotWeights.ckpt"
session.run(tf.global_variables_initializer())


for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_to_batches(training_questions, training_answers, batchSize)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                                targets: padded_answers_in_batch,
                                                                                                lr: learning_rate,
                                                                                                sequence_length: padded_answers_in_batch.shape[1],
                                                                                                keepProb: keep_probability})
        total_training_loss_eror += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batchIndex_check_trainingLoss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training loss error: {:>6.3f}, Training time on 100 batches: {:d} seconds'.format(epoch,
                                                                                                                                        epochs,
                                                                                                                                        batch_index,
                                                                                                                                        len(training_questions) // batchSize,
                                                                                                                                        totalTraining_lossError / batchIndex_check_trainingLoss,
                                                                                                                                        int(batch_time * batchIndex_check_trainingLoss)))
            totalTraining_lossError = 0
        if batch_index % batchIndex_check_validationLoss == 0 & batch_index > 0:
            totalValidaion_lossError = 0
            starting_time = time.time()

        

'''
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                            targets: padded_answers_in_batch,
                                                                                            lr: learning_rate,
                                                                                            sequence_length: padded_answers_in_batch.shape[1],
                                                                                            keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                    epochs,
                                                                                                                                    batch_index,
                                                                                                                                    len(training_questions) // batch_size,
                                                                                                                                    total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                    int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                    targets: padded_answers_in_batch,
                                                                    lr: learning_rate,
                                                                    sequence_length: padded_answers_in_batch.shape[1],
                                                                    keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")
'''
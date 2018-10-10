#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Input, merge, TimeDistributed, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.models import load_model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
import json, argparse, os, pickle
import functools
import re
import io
import sys
from keras.layers.core import *
from keras.models import *
import normalise
#from symspellpy.symspellpy import SymSpell, Verbosity
#from pycontractions import Contractions

import emoji
import unicodedata

print("Importing done")
# In[2]:


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


# In[3]:


def fix_symspell():
    # create object
    initial_capacity = 83000
    # maximum edit distance per dictionary precalculation
    max_edit_distance_dictionary = 2
    prefix_length = 7
    sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,
                         prefix_length)
    # load dictionary
    dictionary_path = "data/frequency_dictionary_en_82_765.txt"
    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")
        return

    # lookup suggestions for single-word input strings
    input_term = "i'm"  # misspelling of "members"
    # max edit distance per lookup
    # (max_edit_distance_lookup <= max_edit_distance_dictionary)
    max_edit_distance_lookup = 2
    suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
    suggestions = sym_spell.lookup(input_term, suggestion_verbosity,
                                   max_edit_distance_lookup)
    # display suggestion term, term frequency, and edit distance
    for suggestion in suggestions:
        print("{}, {}, {}".format(suggestion.term, suggestion.count,
                                  suggestion.distance))

    # lookup suggestions for multi-word input strings (supports compound
    # splitting & merging)
    input_term = ("don't worry i'm girl <eos> hmm how do i know if you are <eos> what's ur name ?")
    # max edit distance per lookup (per single word, not per whole input string)
    max_edit_distance_lookup = 2
    suggestions = sym_spell.lookup_compound(input_term,
                                            max_edit_distance_lookup)
    # display suggestion term, edit distance, and term frequency
    for suggestion in suggestions:
        print("{}, {}, {}".format(suggestion.term, suggestion.count,
                                  suggestion.distance))


# In[4]:


contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


# In[5]:


config = {
          "train_data_path" : "data/train.txt",
          "test_data_path" : "data/devwithoutlabels.txt",
          "solution_path" : "data/test.txt",
          "embedding_matrix_path" : "data/embedding_fasttext.npy",
          "emoji_dict_path" : "data/emoji_dict.txt",
          "fast_text_embedding_path" : "data/wiki-news-300d-1M.vec",
          "glove_embedding_path" : "data/glove.6B.100d.txt",
          "num_folds" : 5,
          "num_classes" : 4,
          "max_nb_words" : 20000,
          "max_sequence_length" : 100,
          "max_charsequence_length": 15,
          "embedding_dim" : 300,
          "batch_size" : 256,
          "lstm_dim" : 128,
          "learning_rate" : 0.01,
          "dropout" : 0.4,
          "num_epochs" : 30
        }


# In[6]:


trainDataPath = config["train_data_path"]
testDataPath = config["test_data_path"]
solutionPath = config["solution_path"]
embeddingMatrixPath = config["embedding_matrix_path"]
emojiDictPath = config["emoji_dict_path"]
fastTextEmbeddingPath = config["fast_text_embedding_path"]
gloveEmbeddingPath = config["glove_embedding_path"]

NUM_FOLDS = config["num_folds"]
NUM_CLASSES = config["num_classes"]
MAX_NB_WORDS = config["max_nb_words"]
MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
MAX_CHARSEQUENCE_LENGTH = config["max_charsequence_length"]
EMBEDDING_DIM = config["embedding_dim"]
BATCH_SIZE = config["batch_size"]
LSTM_DIM = config["lstm_dim"]
DROPOUT = config["dropout"]
LEARNING_RATE = config["learning_rate"]
NUM_EPOCHS = config["num_epochs"]


# In[7]:


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            #print(line)
            count = 0
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '    
                line = cSpace.join(lineSplit)
            
            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)
            
            emoji_dict = json.load(open(emojiDictPath))         
       
            # Expand abbreviations, 
            full_expanded = []
            for c in line[1:4]:
                #replace emojis
                d = c
                for character in c:
                    if character in emoji.UNICODE_EMOJI:
                        #print(c)
                        uni = 'U+' + hex(ord(character))[2:].upper()
                        d = d.replace(character, ' '+emoji_dict[uni]+' ')
                
                expanded = []
                words = d.split()
                for word in words:
                    word = word.replace("’","'")  # difference in apostrophe's
                    ex = contractions.get(word.lower())
                    if not ex:
                        ex = word
                    expanded.append(ex)
                full_expanded.append(' '.join(expanded))
            
            conv = ' <eos> '.join(full_expanded)
            #print(conv)
            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)
            
            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


# In[8]:


print("Processing training data...")
trainIndices, trainTexts, labels_pre = preprocessData(trainDataPath, mode="train")
# Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable   
# writeNormalisedData(trainDataPath, trainTexts)
print("Processing test data...")
testIndices, testTexts = preprocessData(testDataPath, mode="test")
# writeNormalisedData(testDataPath, testTexts)


# In[9]:


def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    #with io.open(gloveEmbedidngPath), encoding="utf8") as f:\
    with io.open((fastTextEmbeddingPath), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector
    
    print('Found %s word vectors.' % len(embeddingsIndex))
    
    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
    
    return embeddingMatrix
            


# In[10]:


print("Extracting tokens...")
tokenizer_word = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer_word.fit_on_texts(trainTexts)
trainSequences = tokenizer_word.texts_to_sequences(trainTexts)
testSequences = tokenizer_word.texts_to_sequences(testTexts)

wordIndex = tokenizer_word.word_index
print("Found %s unique tokens." % len(wordIndex))

print("Populating embedding matrix...")
# embeddingMatrix = getEmbeddingMatrix(wordIndex)

# t = np.where(~embeddingMatrix.any(axis=1))[0]
# np.save(embeddingMatrixPath, embeddingMatrix)
embeddingMatrix = np.load(embeddingMatrixPath)


# In[11]:


print("Extracting tokens/ characters...")
tokenizer_char = Tokenizer(num_words=MAX_NB_WORDS, char_level=True)
tokenizer_char.fit_on_texts(trainTexts)
charIndex = tokenizer_char.word_index
charIndex['PAD'] = 0
charIndex['UNK'] = len(charIndex)
trainSequences_char = []
testSequences_char = []

for s in trainTexts:
    sentence = []
    for w in s.split(' '):
        #if w is not '<eos>':
        word = [charIndex[x] for x in w]
        sentence.append(word)
    trainSequences_char.append(sentence)

for s in testTexts:
    sentence = []
    for w in s.split(' '):
        #if w is not '<eos>':
        word = [charIndex.get(x, charIndex['UNK']) for x in w]
        sentence.append(word)
    testSequences_char.append(sentence)

print("Found %s unique charactertokens." % len(charIndex))


# In[ ]:


# typos = []
# for _ in t:
#     for k, v in wordIndex.items():
#         if _ == v:
#               typos.append(k)


# In[ ]:


# len(typos)


# In[ ]:


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


# In[ ]:


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')    
                except:
                    # If label information not available (test time)
                    fout.write('\n')


# In[ ]:


def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


# In[ ]:


def buildModel(char_output_dim = 10,
               char_dropout=0.5,
               char_lstm_units=40, 
               main_lstm_units = LSTM_DIM, 
               main_dropout = DROPOUT):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    word_in = Input(shape=(MAX_SEQUENCE_LENGTH,))

    
    word_embedding = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                #input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(word_in)
    
    n_chars = len(charIndex)
    # input and embeddings for characters
    char_in = Input(shape=(MAX_SEQUENCE_LENGTH, MAX_CHARSEQUENCE_LENGTH,))
    emb_char = TimeDistributed(Embedding(input_dim=n_chars,
                                         output_dim=char_output_dim,
                                         input_length=MAX_CHARSEQUENCE_LENGTH,
                                         mask_zero=True))(char_in)
    
    # character LSTM to get word encodings by characters
    char_embedding = TimeDistributed(LSTM(units=char_lstm_units, return_sequences=False,
                                    recurrent_dropout=char_dropout))(emb_char)

    
    embeddingLayer = concatenate([word_embedding, char_embedding])
    
    lstmLayer = LSTM(units = main_lstm_units, dropout=main_dropout)(embeddingLayer)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(lstmLayer)
    
    model = Model(inputs=[word_in, char_in], outputs=predictions)
    
#     model = Sequential()
#     model.add(embeddingLayer)
#     model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
#     model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    
    model.summary()

    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc', precision, recall])
    return model


# In[ ]:


def buildModelbase():
    word_in = Input(shape=(MAX_SEQUENCE_LENGTH,))

    
    word_embedding = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                #input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(word_in)
    
    lstmLayer = LSTM(LSTM_DIM, dropout=DROPOUT)(word_embedding)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(lstmLayer)
    
    model = Model(inputs=word_in, outputs=predictions)
    
    model.summary()

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model


# In[ ]:


buildModel()


# In[ ]:


# TIME_STEPS = 100
# INPUT_DIM = 100
# SINGLE_ATTENTION_VECTOR = False

# def attention_3d_block(inputs):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[2])
#     a = Permute((2, 1))(inputs)
#     a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(TIME_STEPS, activation='softmax')(a)
#     if SINGLE_ATTENTION_VECTOR:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((2, 1), name='attention_vec')(a)
#     output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     return output_attention_mul

# def buildModelAttention(embeddingMatrix):
#     """Constructs the architecture of the model
#     Input:
#         embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
#     Output:
#         model : A basic LSTM model
#     """
#     embeddingLayer = Embedding(embeddingMatrix.shape[0],
#                                EMBEDDING_DIM,
#                                 weights=[embeddingMatrix],
#                                 input_length=MAX_SEQUENCE_LENGTH,
#                                 trainable=False)
#     model = Sequential()
#     model.add(embeddingLayer)
# 	#model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
#     #model.add(Dense(NUM_CLASSES, activation='sigmoid'))

#     #inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
#     #inputs = tf.convert_to_tensor(embeddingMatrix, np.float32)

#     #lstm_out = LSTM(LSTM_DIM, return_sequences=True)(inputs)
#     model.add(Lambda(attention_3d_block))

#     model.add(LSTM(LSTM_DIM, return_sequences=True))
#     model.add(Lambda(attention_3d_block))

#  	#attention_mul = attention_3d_block(lstm_out)
#     #attention_mul = Flatten()(attention_mul)
#     model.add(Flatten())
#     #output = Dense(4, activation='sigmoid')(attention_mul)
#     model.add(Dense(4, activation='sigmoid'))
#     #model = Model(input=[embeddingLayer], output=output)

#     model.summary()

#     rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=rmsprop,
#                   metrics=['acc'])

#     return model


# In[ ]:


# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.distplot([len(i) for i in wordIndex.keys()])


# In[ ]:





# In[ ]:


def create_pad_sequences(sequences, sequences_char):
    wordseq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,
                             padding='post', truncating='post')
    charseq_pre = []
    for seq in sequences_char:
        padded_seq = pad_sequences(seq, maxlen=MAX_CHARSEQUENCE_LENGTH,
                                   padding='post', truncating='post')
        charseq_pre.append(padded_seq)
    charseq_pre = np.array(charseq_pre)
    charseq = pad_sequences(charseq_pre, maxlen=MAX_SEQUENCE_LENGTH,
                                 padding='post', truncating='post')
    return [wordseq, charseq]


# In[ ]:


def grid_search(data_train, labels_train, param_grid, fit_params):
    char_output_dim = param_grid['char_output_dim']
    char_lstm_units = param_grid['char_lstm_units']
    main_lstm_units = param_grid['main_lstm_units']
    
    batch_size = fit_params['batch_size']
    epochs = fit_params['epochs']
    validation_data = fit_params['validation_data']
    
    histories = {}
#     cod = char_output_dim[0]
#     clu = char_lstm_units[0]
#     mlu = main_lstm_units[0]
    for cod in char_output_dim:
        for clu in char_lstm_units:
            for mlu in main_lstm_units:
                model = buildModel(char_output_dim=cod,
                                   char_lstm_units=clu,
                                   main_lstm_units=mlu)
                currentConfig = 'cod_' + str(cod) + '_clu_' + str(clu) + '_mlu_' + str(mlu)
                print('Training Configuration:',  currentConfig, flush = True)
                history = model.fit(data_train, labels_train, 
                                    epochs = epochs,
                                    batch_size = batch_size,
                                    validation_data = validation_data,
				    verbose = 2)
                histories[currentConfig] = history.history
                with open('grid_search_histories.json', 'w') as file:
                    file.write(json.dumps(histories))
    return histories
                


# In[ ]:


[data_wordseq,data_charseq] = create_pad_sequences(trainSequences, trainSequences_char)
labels = to_categorical(np.asarray(labels_pre))

# Randomize data
np.random.seed(10)
np.random.shuffle(trainIndices)
lim = int(0.85*len(trainSequences))
data_wordseq_train = data_wordseq[trainIndices][:lim]
data_charseq_train = data_charseq[trainIndices][:lim]
labels_train = labels[trainIndices][:lim]

data_wordseq_valid = data_wordseq[trainIndices][lim:]
data_charseq_valid = data_charseq[trainIndices][lim:]
labels_valid = labels[trainIndices][lim:]

data_train = [data_wordseq_train, data_charseq_train]
data_valid = [data_wordseq_valid, data_charseq_valid]

# data_train = data_wordseq_train
# data_valid = data_wordseq_valid

print("Shape of training data tensor: ", data_train[0].shape, data_train[1].shape)
print("Shape of validation data tensor: ", data_valid[0].shape, data_valid[1].shape)
print("Shape of label tensor: ", labels_train.shape, labels_valid.shape)
'''
# Perform k-fold cross validation
metrics = {"accuracy" : [],
           "microPrecision" : [],
           "microRecall" : [],
           "microF1" : []}

print("Starting k-fold cross validation...")
for k in range(NUM_FOLDS):
    print('-'*40)
    print("Fold %d/%d" % (k+1, NUM_FOLDS))
    validationSize = int(len(data)/NUM_FOLDS)
    index1 = validationSize * k
    index2 = validationSize * (k+1)

    xTrain = np.vstack((data[:index1],data[index2:]))
    yTrain = np.vstack((labels[:index1],labels[index2:]))
    xVal = data[index1:index2]
    yVal = labels[index1:index2]
    print("Building model...")
    model = buildModel(embeddingMatrix)
    model.fit(xTrain, yTrain, 
              validation_data=(xVal, yVal),
              epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    predictions = model.predict(xVal, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
    metrics["accuracy"].append(accuracy)
    metrics["microPrecision"].append(microPrecision)
    metrics["microRecall"].append(microRecall)
    metrics["microF1"].append(microF1)

print("\n============= Metrics =================")
print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"])/len(metrics["accuracy"])))
print("Average Cross-Validation Micro Precision : %.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))

print("\n======================================")
'''
print("Retraining model on entire data to create solution file")

# define the grid search parameters
char_output_dim = [10, 15, 20]
char_lstm_units=[24, 32, 48, 64]
main_lstm_units = [128, 160, 192]

param_grid = dict(char_output_dim = char_output_dim, 
                  char_lstm_units = char_lstm_units, 
                  main_lstm_units = main_lstm_units)
fit_params = dict(batch_size = 64,
                  epochs = 50,
                  validation_data=(data_valid, labels_valid))

histories = grid_search(data_train=data_train,
                        labels_train = labels_train,
                        param_grid=param_grid,
                        fit_params=fit_params)

# model = buildModel(embeddingMatrix)
# history = []
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=2, verbose=1),
#     ModelCheckpoint(filepath='/tmp/weights.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
# ]
# for i in range(5):
# history.append(model.fit(data_train, labels_train,
#                          epochs=int(NUM_EPOCHS),
#                          batch_size=BATCH_SIZE,
#                          #callbacks = callbacks,
#                          validation_data=(data_valid, labels_valid)
#                         ))
#model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
#model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(20, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

# print("Creating solution file...")
# testData = create_pad_sequences(testSequences, testSequences_char)
# predictions = model.predict(testData[0], batch_size=BATCH_SIZE)
# predictions = predictions.argmax(axis=1)

# with io.open(solutionPath, "w", encoding="utf8") as fout:
#     fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
#     with io.open(testDataPath, encoding="utf8") as fin:
#         fin.readline()
#         for lineNum, line in enumerate(fin):
#             fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
#             fout.write(label2emotion[predictions[lineNum]] + '\n')
# print("Completed. Model parameters: ")
# print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" 
#       % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))

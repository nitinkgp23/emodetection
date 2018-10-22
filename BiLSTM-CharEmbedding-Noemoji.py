
# coding: utf-8

# In[24]:


import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Input, merge, TimeDistributed, concatenate, Bidirectional
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

import emoji
import unicodedata


# In[3]:


config = {
          "train_data_path" : "data/train.txt",
          "test_data_path" : "data/devwithoutlabels.txt",
          "solution_path" : "test.txt",
          "embedding_matrix_path" : "data/embedding_conceptnet.npy",   ## change this accordingly
          "emoji_dict_path" : "data/emoji_dict.txt",
          "fast_text_embedding_path" : "data/wiki-news-300d-1M.vec",
          "glove_embedding_path" : "data/glove.6B.100d.txt",
          "conceptnet_path" : "data/conceptnetembed.txt",
          "contractions_path": 'data/contractions.json',
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
          "num_epochs" : 20
        }


# In[4]:


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


# In[6]:


trainDataPath = config["train_data_path"]
testDataPath = config["test_data_path"]
solutionPath = config["solution_path"]
embeddingMatrixPath = config["embedding_matrix_path"]
emojiDictPath = config["emoji_dict_path"]
fastTextEmbeddingPath = config["fast_text_embedding_path"]
gloveEmbeddingPath = config["glove_embedding_path"]
conceptnetEmbeddingPath = config["conceptnet_path"]
contractionsPath = config["contractions_path"]
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


with open(contractionsPath) as f:
    contractions = json.load(f)


# In[8]:


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
#                 for character in c:
#                     if character in emoji.UNICODE_EMOJI:
#                         #print(c)
#                         uni = 'U+' + hex(ord(character))[2:].upper()
#                         d = d.replace(character, ' '+emoji_dict[uni]+' ')

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


# In[9]:


print("Processing training data...")
trainIndices, trainTexts, labels_pre = preprocessData(trainDataPath, mode="train")
# Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable   
# writeNormalisedData(trainDataPath, trainTexts)
print("Processing test data...")
testIndices, testTexts = preprocessData(testDataPath, mode="test")
# writeNormalisedData(testDataPath, testTexts)


# In[10]:


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
    with io.open((conceptnetEmbeddingPath), encoding="utf8") as f:
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
            


# In[11]:


print("Extracting tokens...")
tokenizer_word = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer_word.fit_on_texts(trainTexts)
trainSequences = tokenizer_word.texts_to_sequences(trainTexts)
testSequences = tokenizer_word.texts_to_sequences(testTexts)

wordIndex = tokenizer_word.word_index
print("Found %s unique tokens." % len(wordIndex))

print("Populating embedding matrix...")
embeddingMatrix = getEmbeddingMatrix(wordIndex)

# t = np.where(~embeddingMatrix.any(axis=1))[0]
np.save(embeddingMatrixPath, embeddingMatrix)
#embeddingMatrix = np.load(embeddingMatrixPath)


# In[12]:


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


# In[13]:


# typos = []
# for _ in t:
#     for k, v in wordIndex.items():
#         if _ == v:
#               typos.append(k)


# In[14]:


# len(typos)


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[28]:


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
    char_embedding = TimeDistributed(Bidirectional(LSTM(units=char_lstm_units, return_sequences=False,
                                    recurrent_dropout=char_dropout)))(emb_char)

    
    embeddingLayer = concatenate([word_embedding, char_embedding])
    
    lstmLayer = Bidirectional(LSTM(units = main_lstm_units, dropout=main_dropout))(embeddingLayer)
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


# In[29]:


def buildModelbase():
    word_in = Input(shape=(MAX_SEQUENCE_LENGTH,))

    
    word_embedding = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                #input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(word_in)
    
    lstmLayer = BiLSTM(LSTM_DIM, dropout=DROPOUT)(word_embedding)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(lstmLayer)
    
    model = Model(inputs=word_in, outputs=predictions)
    
    model.summary()

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model


# In[30]:


buildModel()


# In[22]:


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


# In[23]:


[data_wordseq,data_charseq] = create_pad_sequences(trainSequences, trainSequences_char)
labels = to_categorical(np.asarray(labels_pre))

# Randomize data
np.random.seed(10)
np.random.shuffle(trainIndices)
# lim = int(0.85*len(trainSequences))
data_wordseq = data_wordseq[trainIndices]
data_charseq = data_charseq[trainIndices]
labels_train = labels[trainIndices]

data_train = [data_wordseq, data_charseq]

cod = 15
clu = 48
mlu = 160

# Perform k-fold cross validation
metrics = {"accuracy" : [],
           "microPrecision" : [],
           "microRecall" : [],
           "microF1" : []}

print("Starting k-fold cross validation...")
for k in range(3):
    print('-'*40)
    print("Fold {}/{}".format(k+1, NUM_FOLDS))
    validationSize = int(len(labels_train)/NUM_FOLDS)
    index1 = validationSize * k
    index2 = validationSize * (k+1)

    xTrain_word = np.vstack((data_wordseq[:index1],data_wordseq[index2:]))
    xTrain_char = np.vstack((data_charseq[:index1],data_charseq[index2:]))
    yTrain = np.vstack((labels_train[:index1],labels_train[index2:]))
    xTrain = [xTrain_word, xTrain_char]

    xVal_word = data_wordseq[index1:index2]
    xVal_char = data_charseq[index1:index2]
    yVal = labels_train[index1:index2]
    xVal = [xVal_word, xVal_char]

    
    print("Building model...")
    model = buildModel(char_output_dim=cod,
                       char_lstm_units=clu,
                       main_lstm_units=mlu
                      )
    model.fit(xTrain, yTrain, 
              validation_data=(xVal, yVal),
              epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=2)

    predictions = model.predict(xVal, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
    metrics["accuracy"].append(accuracy)
    metrics["microPrecision"].append(microPrecision)
    metrics["microRecall"].append(microRecall)
    metrics["microF1"].append(microF1)

print("\n============= Metrics =================")
print("Average Cross-Validation Accuracy : ")
print("%.4f" % (sum(metrics["accuracy"])/len(metrics["accuracy"])))
print("Average Cross-Validation Micro Precision :")
print("%.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
print("Average Cross-Validation Micro Recall :")
print("%.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
print("Average Cross-Validation Micro F1 :")
print("%.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))

print("\n======================================")

print("Retraining model on entire data to create solution file")


model = buildModel(char_output_dim=cod,
                   char_lstm_units=clu,
                   main_lstm_units=mlu
                  )
history = model.fit(data_train, labels_train, 
                    epochs = int(NUM_EPOCHS),
                    batch_size = BATCH_SIZE,
                    verbose=2)

#model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
#model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(20, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

print("Creating solution file...")
testData = create_pad_sequences(testSequences, testSequences_char)
predictions = model.predict(testData, batch_size=BATCH_SIZE)
predictions = predictions.argmax(axis=1)

with io.open(solutionPath, "w", encoding="utf8") as fout:
    fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
    with io.open(testDataPath, encoding="utf8") as fin:
        fin.readline()
        for lineNum, line in enumerate(fin):
            fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
            fout.write(label2emotion[predictions[lineNum]] + '\n')
print("Completed. Model parameters: ")
print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" 
      % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))


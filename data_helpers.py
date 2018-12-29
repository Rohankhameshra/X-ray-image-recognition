import numpy as np
import re
import itertools
import csv
from collections import Counter
import os
import pickle
"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    doc1= []
    doc2= []
    doc3 = []
    data=[]
    text = {}
    reader = csv.reader(open("train.csv"), delimiter=',')
    for row in reader:
        data.append(row)
    for linknumber in range(1,112963):   
        title= data[linknumber][1]
        image= data[linknumber][0]
        label= data[linknumber][2]
        #print (title)
        doc1.append(title)
        doc2.append(image)
        if (label=="True"):
            doc3.append(1)
        if (label=="False"):
            doc3.append(0)
    image =list(doc2)    
    product = list(doc1)
    label = list(doc3)
    product = [s.strip() for s in product]
    
    # Split by words
    x_text = product
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    #print (x_text)
    return [x_text],image, label


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences
def pad_sentence(sentence, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    num_padding = 44 - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    #Save vectorizer.vocabulary_
    pickle.dump(vocabulary_inv,open(os.path.join("./model/feature_inv.pkl"),"wb"), protocol=2)
    pickle.dump(vocabulary,open(os.path.join("./model/feature.pkl"),"wb"),  protocol=2)
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return [x]


def build_input_data_for_sentences(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


def load_text_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, _ ,_= load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    #vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    #x = build_input_data(sentences_padded, vocabulary)
    return sentences_padded

def load_image_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    _,image_num,_ = load_data_and_labels()

    return image_num
def load_label_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    _,_,label = load_data_and_labels()

    return label


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at eprinterh epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def my_get_input_sentence(title):
    #raw = input("input a title: ")
    raw = clean_str(title)
    print (raw)
    raw_comment_cut = raw.split()
    print (raw_comment_cut)
    sentence_padded = pad_sentence(raw_comment_cut)
    vocabulary=pickle.load(open(os.path.join(settings.BASE_DIR,"./model/feature.pkl"), "rb"))
    #vocabulary, vocabulary_inv = build_vocab(sentence_padded)
    x = build_input_data_for_sentences(sentence_padded, vocabulary)
    return x



import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from imagenet_utils import preprocess_input
from keras.preprocessing import image as image_utils
from keras import initializers
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from keras import regularizers
import sys
import os
import numpy as np
import data_helpers
from w2v import train_word2vec


from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras import backend as K

from gensim.models import Word2Vec
from gensim import corpora

top_model_weights_path = 'bottleneck_fc_model.h5'
epochs = 50
batch_size = 50
category = 2

min_word_count = 1  # Minimum word count
context = 10        # Context window size
embedding_dim = 20

sentences = data_helpers.load_text_data()
images= data_helpers.load_image_data()

labels=data_helpers.load_label_data()
#embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
#x = embedding_weights[0][x]
#print (x)
img_width, img_height = 224, 224
# build the VGG16 network
img_model = applications.VGG16(include_top=False, weights='imagenet')
#print (sentences[0])
# train model
text_model = Word2Vec(sentences[0], min_count=1, workers=8)
# summarize the loaded model
#print(text_model)
# summarize vocabulary
words = list(text_model.wv.vocab)
#print(words)
train_vector=np.empty([1,100])
# save model
text_model.save('text_model.bin')
train_labels=np.empty([1])
train_data=np.empty([25188,])
positive=0
negative=0
i=0
while(positive<20000 or negative<20000):
    i+=1
    vector=np.zeros([1,100])
    if(labels[i]==1 and labels[i]!=0 and positive<20000):
        for j in sentences[0][i]:
            #print (j)
            vector=np.add(text_model.wv[j],vector)
        #train_vector=np.concatenate((train_vector,vector), axis=1) 
        print (i)
        print (images[i])
        image = image_utils.load_img("/home/rohan/Desktop/x_ray_image_recognition_data/x_ray_images/train_xray_images/%s"%str(images[i]), target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        preds = img_model.predict(image)
        #print (preds.shape)
        #print (vector.shape)
        train_labels= np.append(train_labels,labels[i])
        input_vector=np.append(preds,vector/float(len(sentences[0][i])))
        #print (input_vector.shape)
        train_data=np.vstack((train_data,input_vector))
        positive+=1
        print("positive=%i"%positive)
    if(labels[i]==0 and labels[i]!=1 and negative<20000):
        for j in sentences[0][i]:
            #print (j)
            vector=np.add(text_model.wv[j],vector)
        #train_vector=np.concatenate((train_vector,vector), axis=1) 
        print (i)
        print (images[i])
        image = image_utils.load_img("/home/rohan/Desktop/x_ray_image_recognition_data/x_ray_images/train_xray_images/%s"%str(images[i]), target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        preds = img_model.predict(image)
        #print (preds.shape)
        #print (vector.shape)
        train_labels= np.append(train_labels,labels[i])
        input_vector=np.append(preds,vector/float(len(sentences[0][i])))
        #print (input_vector.shape)
        train_data=np.vstack((train_data,input_vector))
        negative+=1
        print ("negative=%i"%negative)
    
    

 
np.save(open('bottleneck_features_train.npy', 'wb'),train_data)
np.save(open('bottleneck_labels_train.npy', 'wb'),train_labels)

validation_labels=np.empty([1])
negative=0
positive=0
validation_data=np.empty([25188,])
while(positive<4000 or negative<4000):
    i+=1
    vector=np.zeros([1,100])
    if(labels[i]==1 and labels[i]!=0 and positive <4000):
        for j in sentences[0][i]:
            #print (j)
            vector=np.add(text_model.wv[j],vector)
        #train_vector=np.concatenate((train_vector,vector), axis=1) 
        print (i)
        print (images[i])
        image = image_utils.load_img("/home/rohan/Desktop/x_ray_image_recognition_data/x_ray_images/train_xray_images/%s"%str(images[i]), target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        preds = img_model.predict(image)
        #print (preds.shape)
        #print (vector.shape)
        validation_labels= np.append(validation_labels,labels[i])
        input_vector=np.append(preds,vector/float(len(sentences[0][i])))
        #print (input_vector.shape)
        validation_data=np.vstack((validation_data,input_vector))
        positive+=1
        print("positive=%i"%positive)
    if(labels[i]==0 and labels[i]!=1 and negative<4000):
        for j in sentences[0][i]:
            #print (j)
            vector=np.add(text_model.wv[j],vector)
        #train_vector=np.concatenate((train_vector,vector), axis=1) 
        print (i)
        print (images[i])
        image = image_utils.load_img("/home/rohan/Desktop/x_ray_image_recognition_data/x_ray_images/train_xray_images/%s"%str(images[i]), target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        preds = img_model.predict(image)
        #print (preds.shape)
        #print (vector.shape)
        validation_labels= np.append(validation_labels,labels[i])
        input_vector=np.append(preds,vector/float(len(sentences[0][i])))
        #print (input_vector.shape)
        validation_data=np.vstack((validation_data,input_vector))
        negative+=1
        print ("negative=%i"%negative)
    
    

np.save(open('bottleneck_features_validation.npy', 'wb'),validation_data)
np.save(open('bottleneck_labels_validation.npy', 'wb'),validation_labels)

train_data = np.load(open('bottleneck_features_train.npy','rb'))
validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
#labels=data_helpers.load_label_data()
train_labels = np.load(open('bottleneck_labels_train.npy','rb'))
validation_labels = np.load(open('bottleneck_labels_validation.npy','rb'))
model = Sequential()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data)
X_validation_scaled=scaler.fit_transform(validation_data)
print (list(train_data.shape[1:])[0])
#model.add(Flatten(input_shape=(1, list(train_data.shape[1:])[0], )))
model.add(Dense(256,input_shape=train_data.shape[1:], activation='relu',kernel_initializer=initializers.glorot_uniform(seed = None)))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
hist = model.fit(X_train_scaled, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_validation_scaled, validation_labels))
model.save_weights(top_model_weights_path)

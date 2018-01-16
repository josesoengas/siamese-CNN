from __future__ import division, print_function
from keras import backend as K
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.optimizers import Adam


from keras.models import load_model

VECTOR_SIZE = 2048
BATCH_SIZE=1
'''
demo_1 = [('5.jpg','6.jpg',0),
('The-Procuress-by-Dirck-van-Baburen.jpg','The-Denial-of-Saint-Peter-by-Dirck-van-Baburen.jpg',0),
('Jan_Vermeer_-_The_Art_of_Painting_-_Google_Art_Project.jpg','Johannes_Vermeer_-_De_Soldaat_en_het_Lachende_Meisje_-_Google_Art_Project.jpg',1),
('the-milkmaid-reproduction-original-by-johannes-vermeer-benita-mulokaite.jpg','5.jpg',1),
('Lady w Blue Hat.jpg','girl_with_a_pearl_earring.jpg',0),
]
'''

demo_0 = [('5.jpg','6.jpg',0)]
demo_1 = [('The-Procuress-by-Dirck-van-Baburen.jpg','The-Denial-of-Saint-Peter-by-Dirck-van-Baburen.jpg',0)]
demo_2 = [('Jan_Vermeer_-_The_Art_of_Painting_-_Google_Art_Project.jpg','Johannes_Vermeer_-_De_Soldaat_en_het_Lachende_Meisje_-_Google_Art_Project.jpg',1)]
demo_3 = [('the-milkmaid-reproduction-original-by-johannes-vermeer-benita-mulokaite.jpg','5.jpg',1)]
demo_4 = [('Lady w Blue Hat.jpg','girl_with_a_pearl_earring.jpg',0)]
demo_5 = [('7.jpg','8.jpg',0)]
demo_6 = [('Lady w Blue Hat.jpg','Vermeer_-_Girl_with_a_Red_Hat.JPG',0)]
demo_7 = [('girl_with_a_pearl_earring.jpg','Vermeer_-_Girl_with_a_Red_Hat.JPG',1)]



demo_list = [demo_0,demo_1,demo_2,demo_3,demo_4,demo_5,demo_6,demo_7]


def load_vectors(vector_file):
        vec_dict = {}
        fvec = open(vector_file, "r")
        for line in fvec:
                image_name, image_vec = line.strip().split("\t")
                vec = np.array([float(v) for v in image_vec.split(",")])
                vec_dict[image_name] = vec
        fvec.close()
        return vec_dict

def batch_to_vectors(batch, vec_size, vec_dict):
        X1 = np.zeros((len(batch), vec_size))
        X2 = np.zeros((len(batch), vec_size))
        Y = np.zeros((len(batch), 2))
        for tid in range(len(batch)):
                X1[tid] = vec_dict[batch[tid][0]]
                X2[tid] = vec_dict[batch[tid][1]]
                Y[tid] = [1, 0] if batch[tid][2] == 0 else [0, 1]
        return ([X1, X2], Y)

def data_generator(triples, vec_size, vec_dict, batch_size=32):
        while True:
                # shuffle once per batch
                indices = np.random.permutation(np.arange(len(triples)))
                num_batches = len(triples) // batch_size
                for bid in range(num_batches):
                        batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
                        batch = [triples[i] for i in batch_indices]
                        yield batch_to_vectors(batch, vec_size, vec_dict)


def evaluate_model(model, test_gen):
       
        ytrue, ypred = [], []
        num_test_steps = len(demo_1) // BATCH_SIZE
        print(num_test_steps)
        for i in range(num_test_steps):
                (X1, X2), Y = next(test_gen)
                Y_ = model.predict([X1, X2])
                ytrue.extend(np.argmax(Y, axis=1).tolist())
                ypred.extend(np.argmax(Y_, axis=1).tolist())
        accuracy = accuracy_score(ytrue, ypred)
        print('true',str(ytrue))
        print('pred',str(ypred))
        print("\nAccuracy: {:.3f}".format(accuracy))
        print("\nConfusion Matrix")
        print(confusion_matrix(ytrue, ypred))
        print("\nClassification Report")
        print(classification_report(ytrue, ypred))
        return accuracy

# [('1200px-Wheat-Field-with-Cypresses-(1889)-Vincent-van-Gogh-Met.jpg', 
#	'The-Denial-of-Saint-Peter-by-Dirck-van-Baburen.jpg', 0)]


data_drive_1 = '/home/msugimura/ms_code_repository/data_folder/siamese_1/'

VECTOR_FILE = os.path.join(data_drive_1, "inceptionv3_test_1_-vectors.tsv")
vec_dict = load_vectors(VECTOR_FILE)

model_path = '/home/msugimura/ms_code_repository/data_folder/siamese_1/models/inceptionv3-cat-best.h5'


model = load_model(model_path)


model.summary()
for demo in demo_list:

	test_gen = data_generator(demo, VECTOR_SIZE, vec_dict, BATCH_SIZE)
	demo_accuracy = evaluate_model(model, test_gen)

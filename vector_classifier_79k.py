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


BATCH_SIZE = 256
#NUM_EPOCHS = 10
NUM_EPOCHS = 35

data_drive_1 = '/home/msugimura/ms_code_repository/data_folder/siamese_1/'
dat = pd.read_csv('/home/msugimura/ms_code_repository/data_folder/siamese_1/train_76k.csv')

img_dir = '/home/msugimura/ms_code_repository/data_folder/data/train/train/'
#img_list = os.listdir(img_dir)

def get_holiday_triples(image_dir):
        image_groups = {}
        for index, row in dat.iterrows():
            img_name = row['new_filename']
            group_name = row['artist']
            if group_name in image_groups:
                image_groups[group_name].append(img_name)
            else:
                image_groups[group_name] = [img_name]

        num_sims = 0
        image_triples = []
        group_list = sorted(list(image_groups.keys()))
        for i, g in enumerate(group_list):
                if num_sims % 100 == 0:
                        print("Generated {:d} pos + {:d} neg = {:d} total image triples"
                                    .format(num_sims, num_sims, 2*num_sims))
                images_in_group = image_groups[g]
                sim_pairs_it = itertools.combinations(images_in_group, 2)
                # for each similar pair, generate a corresponding different pair
                for ref_image, sim_image in sim_pairs_it:
                        image_triples.append((ref_image, sim_image, 1))
                        num_sims += 1
                        while True:
                                j = np.random.randint(low=0, high=len(group_list), size=1)[0]
                                if j != i:
                                        break
                        dif_image_candidates = image_groups[group_list[j]]
                        k = np.random.randint(low=0, high=len(dif_image_candidates), size=1)[0]
                        dif_image = dif_image_candidates[k]
                        image_triples.append((ref_image, dif_image, 0))
        print("Generated {:d} pos + {:d} neg = {:d} total image triples"
                    .format(num_sims, num_sims, 2*num_sims))
        return image_triples

def load_vectors(vector_file):
        vec_dict = {}
        fvec = open(vector_file, "r")
        for line in fvec:
                image_name, image_vec = line.strip().split("\t")
                vec = np.array([float(v) for v in image_vec.split(",")])
                vec_dict[image_name] = vec
        fvec.close()
        return vec_dict

def train_test_split(triples, splits):
        assert sum(splits) == 1.0
        split_pts = np.cumsum(np.array([0.] + splits))
        indices = np.random.permutation(np.arange(len(triples)))
        shuffled_triples = [triples[i] for i in indices]
        data_splits = []
        for sid in range(len(splits)):
                start = int(split_pts[sid] * len(triples))
                end = int(split_pts[sid + 1] * len(triples))
                data_splits.append(shuffled_triples[start:end])
        return data_splits

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

def evaluate_model(model_file, test_gen):
        model_name = os.path.basename(model_file)
        model = load_model(model_file)
        print("=== Evaluating model: {:s} ===".format(model_name))
        ytrue, ypred = [], []
        num_test_steps = len(test_triples) // BATCH_SIZE
        for i in range(num_test_steps):
                (X1, X2), Y = next(test_gen)
                Y_ = model.predict([X1, X2])
                ytrue.extend(np.argmax(Y, axis=1).tolist())
                ypred.extend(np.argmax(Y_, axis=1).tolist())
        accuracy = accuracy_score(ytrue, ypred)
        print("\nAccuracy: {:.3f}".format(accuracy))
        print("\nConfusion Matrix")
        print(confusion_matrix(ytrue, ypred))
        print("\nClassification Report")
        print(classification_report(ytrue, ypred))
        return accuracy

def get_model_file(data_dir, vector_name, merge_mode, borf):
        return os.path.join(data_dir, "models", "{:s}-{:s}-{:s}.h5"
                                                .format(vector_name, merge_mode, borf))




VECTORIZERS = ["InceptionV3"]
MERGE_MODES = ["Concat", "Euclidean"]
scores = np.zeros((len(VECTORIZERS), len(MERGE_MODES)))


img_dir = '/home/msugimura/ms_code_repository/data_folder/data/train/train/'
#img_list = os.listdir(img_dir)
image_sets = get_holiday_triples(img_dir)

#print(image_sets)

train_triples, val_triples, test_triples = train_test_split(image_sets, splits=[0.8, 0.1, 0.1])
print(len(train_triples), len(val_triples), len(test_triples))


VECTOR_SIZE = 2048
VECTOR_FILE = os.path.join(data_drive_1, "inceptionv3_79k-vectors.tsv")

vec_dict = load_vectors(VECTOR_FILE)

#print(vec_dict)

train_gen = data_generator(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
val_gen = data_generator(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
print('hello world')

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))
merged = Concatenate(axis=-1)([input_1, input_2])



fc1 = Dense(2048, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(0.2)(fc1)
fc1 = Activation("relu")(fc1)

fc2 = Dense(2048, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(0.2)(fc2)
fc2 = Activation("relu")(fc2)


fc3 = Dense(2048, kernel_initializer="glorot_uniform")(fc2)
fc3 = Dropout(0.2)(fc3)
fc3 = Activation("relu")(fc3)


fc8 = Dense(1024, kernel_initializer="glorot_uniform")(fc3)
fc8 = Dropout(0.2)(fc8)
fc8 = Activation("relu")(fc8)


fc9 = Dense(1024, kernel_initializer="glorot_uniform")(fc8)
fc9 = Dropout(0.2)(fc9)
fc9 = Activation("relu")(fc9)

fc11 = Dense(1024, kernel_initializer="glorot_uniform")(fc9)
fc11 = Dropout(0.2)(fc11)
fc11 = Activation("relu")(fc11)

fc12 = Dense(512, kernel_initializer="glorot_uniform")(fc11)
fc12 = Dropout(0.2)(fc12)
fc12 = Activation("relu")(fc12)


fc13 = Dense(512, kernel_initializer="glorot_uniform")(fc12)
fc13 = Dropout(0.2)(fc13)
fc13 = Activation("relu")(fc13)


fc14 = Dense(128, kernel_initializer="glorot_uniform")(fc13)
fc14 = Dropout(0.2)(fc14)
fc14 = Activation("relu")(fc14)

fc15 = Dense(128, kernel_initializer="glorot_uniform")(fc14)
fc15 = Dropout(0.2)(fc15)
fc15 = Activation("relu")(fc15)


pred = Dense(2, kernel_initializer="glorot_uniform")(fc15)
pred = Activation("softmax")(pred)

model = Model(inputs=[input_1, input_2], outputs=pred)

#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adam = Adam(lr=.00001)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
print('concatnated model')
best_model_name = get_model_file(data_drive_1, "inceptionv3", "cat", "best")
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)
train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE
history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, 
                              epochs=NUM_EPOCHS, 
                              validation_data=val_gen, validation_steps=val_steps_per_epoch,
                              callbacks=[checkpoint])

final_model_name = get_model_file(data_drive_1, "inceptionv3", "cat", "final")
model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[0, 0] = best_accuracy if best_accuracy > final_accuracy else final_accuracy

#kept euclidean distance for testing but was found to be not effective on dataset 
'''
### EUCLIDEAN DISTANCE
print('euclidean distance')
train_gen = data_generator(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
val_gen = data_generator(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)

def euclidean_distance(vecs):
    x, y = vecs
    return K.sqrt(K.sum(K.stack([K.square(x), -K.square(y)], axis=1), axis=1))

def euclidean_distance_output_shape(shapes):
    xshape, yshape = shapes
    return xshape

vecs = [np.random.random((10,)), np.random.random((10,))]
print(vecs[0].shape, vecs[1].shape)
s = euclidean_distance(vecs)
print(s.shape)

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))
merged = Lambda(euclidean_distance, 
                output_shape=euclidean_distance_output_shape)([input_1, input_2])



fc1 = Dense(3000, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(0.2)(fc1)
fc1 = Activation("relu")(fc1)

fc2 = Dense(3000, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(0.2)(fc2)
fc2 = Activation("relu")(fc2)

fc4 = Dense(2048, kernel_initializer="glorot_uniform")(fc2)
fc4 = Dropout(0.2)(fc4)
fc4 = Activation("relu")(fc4)

fc5 = Dense(2048, kernel_initializer="glorot_uniform")(fc4)
fc5 = Dropout(0.2)(fc5)
fc5 = Activation("relu")(fc5)

fc6 = Dense(2048, kernel_initializer="glorot_uniform")(fc5)
fc6 = Dropout(0.2)(fc6)
fc6 = Activation("relu")(fc6)


fc8 = Dense(1024, kernel_initializer="glorot_uniform")(fc6)
fc8 = Dropout(0.2)(fc8)
fc8 = Activation("relu")(fc8)


fc9 = Dense(1024, kernel_initializer="glorot_uniform")(fc8)
fc9 = Dropout(0.2)(fc9)
fc9 = Activation("relu")(fc9)

fc11 = Dense(1024, kernel_initializer="glorot_uniform")(fc9)
fc11 = Dropout(0.2)(fc11)
fc11 = Activation("relu")(fc11)

fc12 = Dense(512, kernel_initializer="glorot_uniform")(fc11)
fc12 = Dropout(0.2)(fc12)
fc12 = Activation("relu")(fc12)

fc13 = Dense(512, kernel_initializer="glorot_uniform")(fc12)
fc13 = Dropout(0.2)(fc13)
fc13 = Activation("relu")(fc13)


fc14 = Dense(128, kernel_initializer="glorot_uniform")(fc13)
fc14 = Dropout(0.2)(fc14)
fc14 = Activation("relu")(fc14)


fc15 = Dense(64, kernel_initializer="glorot_uniform")(fc14)
fc15 = Dropout(0.2)(fc15)
fc15 = Activation("relu")(fc15)


pred = Dense(2, kernel_initializer="glorot_uniform")(fc15)
pred = Activation("softmax")(pred)


model = Model(inputs=[input_1, input_2], outputs=pred)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

best_model_name = get_model_file(data_drive_1, "inceptionv3", "l2", "best")
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)
train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE
history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, 
                              epochs=NUM_EPOCHS, 
                              validation_data=val_gen, validation_steps=val_steps_per_epoch,
                              callbacks=[checkpoint])


final_model_name = get_model_file(data_drive_1, "inceptionv3", "l2", "final")
model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[0, 1] = best_accuracy if best_accuracy > final_accuracy else final_accuracy

'''

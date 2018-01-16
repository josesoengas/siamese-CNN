from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
#from keras.applications import vgg16, vgg19, inception_v3, resnet50, xception

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import os
import pandas as pd

##image bomb? 
from PIL.Image import LANCZOS
from PIL import Image
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = 1000000000                                                                                              
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = VGG19(weights='imagenet',include_top=False,pooling='avg')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
dat = pd.read_csv('/home/msugimura/ms_code_repository/data_folder/siamese_1/train_76k.csv')

#dat['img_name'] = 'C:/Users/585000/Desktop/AvantGuard/train/'+ dat['new_filename']

img_list = dat['new_filename'].tolist()
img_dir = '/home/msugimura/ms_code_repository/data_folder/data/train/train/'

#img_list = os.listdir(img_dir)
print(len(img_list))

##### build the inception vectors
model = InceptionV3(weights='imagenet',include_top=False,pooling='avg')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

fvec = open('inceptionv3_79k-vectors.tsv', "w")
num_vecs = 0 
for image_ in img_list:
	img = image.load_img(img_dir+image_, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = model.predict(x)[0]
	# convert from Numpy to a list of values
	#features_arr = np.char.mod('%f', features)

	if num_vecs % 100 == 0:
		print("{:d} vectors generated".format(num_vecs))

	image_vector = ",".join(["{:.5e}".format(v) for v in features.tolist()])
	fvec.write("{:s}\t{:s}\n".format(image_, image_vector))
	num_vecs += 1


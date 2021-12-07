#!/usr/bin/env python
# coding: utf-8
# SSD7 Validation Tutorial (Recuerda ALDRICH ESTE ES EL DE SIEMPRE (PATRONES, ALFAPILOT, TMR)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from time import time

from models.keras_ssd7 import build_model
from keras.utils import plot_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_data_generator import DataGenerator
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes

import glob

img_height = 240 # Height of the input images alp=180
img_width = 320 # Width of the input images	alp=270
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 3 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size

K.clear_session() # Clear previous models from memory.
model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# Load some weights
#model.load_weights('ssd7_model.h5', by_name=True)

# Adam optimizer and the SSD loss function and compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = '/home/oyuki/Downloads/deep_learning/ssd_keras/ssd7_model_flagAndplat2.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes, 
                        'compute_loss': ssd_loss.compute_loss})

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# Images
images_dir = '/home/oyuki/Downloads/deep_learning/AutonomousLanding_MAVs/dataset/imav_flag/images/'

# Ground truth
train_labels_filename = '/home/oyuki/Downloads/deep_learning/AutonomousLanding_MAVs/dataset/imav_flag/labels_train.csv'
val_labels_filename   = '/home/oyuki/Downloads/deep_learning/AutonomousLanding_MAVs/dataset/imav_flag/labels_val.csv'

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)

########## Make predictions ############
# 1: Set the generator for the predictions.
predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)
                                         
# 2: Generate samples
batch_images, batch_labels, batch_filenames = next(predict_generator)

i = 0 # Which batch item to look at
#print("Image:", batch_filenames[i])
print("\nGround truth boxes:")
print(batch_labels[i])





path = "/home/oyuki/images/validation/*.*"
for file in glob.glob(path):
	#print(file)
	img1 = cv2.imread(file)
	print(file)

	img1 = cv2.resize(img1, (320, 240))
	img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
	img3 = np.expand_dims(np.asarray(img2),axis=0)

	# 3: Make a prediction
	time1 = time()
	y_pred = model.predict(img3)
	#print(model.summary())

	trans = time() - time1
	#print("\nTime:")
	#print(trans)

	# 4: Decode the raw prediction `y_pred`
	y_pred_decoded = decode_detections(y_pred,
									   confidence_thresh = 0.5, #0.5, 0.1
									   iou_threshold= 0.45,#0.45, 0.1
									   top_k = 'all', # all 200,		# top highest confidence boxes are returned
									   input_coords='centroids',
									   normalize_coords=normalize_coords,
									   img_height=img_height,
									   img_width=img_width)

	np.set_printoptions(precision=2, suppress=True, linewidth=90)
	fps = 'FPS: {:.1f}',format(1/trans)
	#print("\nPredicted boxes:")
	#print('   class   conf xmin   ymin   xmax   ymax')
	#print(fps)
	classes = ['flag', 'flag', 'no_flag', 'platform'] # Just so we can print class names onto the image instead of IDs

	predicted_boxes = y_pred_decoded[0]
	print(y_pred_decoded[0])

	# 5: Draw the predicted boxes onto the image
	cx = 0.0
	cy = 0.0
	#for box in y_pred_decoded[i]:
	for box in predicted_boxes[:]:
	  #if box[0] == 3.0:				#remember change this (class id test)
		xmin = int(box[-4])
		ymin = int(box[-3])
		xmax = int(box[-2])
		ymax = int(box[-1])
		label = '{}: {:.1f}'.format(classes[int(box[0])], box[1])

		cx = int((xmax + xmin)/ 2)
		cy = int((ymax + ymin)/ 2)

	  #else:
	   # xmin1 = -1
		#ymin1 = -1
		#xmax1 = -1
		#ymax1 = -1

	cv2.rectangle(img1, (int(xmin), int(ymin+5)), (int(xmax), int(ymax)), (0,255,0), 2)
	cv2.circle(img1, (cx, cy), 3, (0, 255, 0), -1)
	cv2.putText(img1, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
	cv2.imshow('Validation', img1)
	cv2.waitKey(1)
	#cv2.destroyAllWindows()

######## To one picture ########

#~ img1 = cv2.imread('/home/oyuki/images/validation/image_0.jpg', cv2.IMREAD_COLOR) #2926 17
#~ img1 = cv2.resize(img1, (320, 240))
#~ img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#~ img3 = np.expand_dims(np.asarray(img2),axis=0)

#~ # 3: Make a prediction
#~ time1 = time()
#~ y_pred = model.predict(img3)

#~ #print(model.summary())

#~ trans = time() - time1
#~ print("\nTime:")
#~ print(trans)

#~ # 4: Decode the raw prediction `y_pred`
#~ y_pred_decoded = decode_detections(y_pred,
                                   #~ confidence_thresh = 0.5, #0.5, 0.1
                                   #~ iou_threshold= 0.45,#0.45, 0.1
                                   #~ top_k = 'all', # all 200,		# top highest confidence boxes are returned
                                   #~ input_coords='centroids',
                                   #~ normalize_coords=normalize_coords,
                                   #~ img_height=img_height,
                                   #~ img_width=img_width)

#~ np.set_printoptions(precision=2, suppress=True, linewidth=90)
#~ fps = 'FPS: {:.1f}',format(1/trans)
#~ print("\nPredicted boxes:")
#~ print('   class   conf xmin   ymin   xmax   ymax')
#~ #print(fps)
#~ classes = ['flag', 'flag', 'no_flag', 'platform'] # Just so we can print class names onto the image instead of IDs

#~ predicted_boxes = y_pred_decoded[0]
#~ print(y_pred_decoded[0])

#~ # 5: Draw the predicted boxes onto the image
#~ cx = 0.0
#~ cy = 0.0
#~ #for box in y_pred_decoded[i]:
#~ for box in predicted_boxes[:]:
  #~ #if box[0] == 3.0:				#remember change this (class id test)
    #~ xmin = int(box[-4])
    #~ ymin = int(box[-3])
    #~ xmax = int(box[-2])
    #~ ymax = int(box[-1])
    #~ label = '{}: {:.1f}'.format(classes[int(box[0])], box[1])

    #~ cx = int((xmax + xmin)/ 2)
    #~ cy = int((ymax + ymin)/ 2)

  #~ #else:
   #~ # xmin1 = -1
    #~ #ymin1 = -1
    #~ #xmax1 = -1
    #~ #ymax1 = -1

#~ cv2.rectangle(img1, (int(xmin), int(ymin+5)), (int(xmax), int(ymax)), (0,255,0), 2)
#~ cv2.circle(img1, (cx, cy), 3, (0, 255, 0), -1)
#~ cv2.putText(img1, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#~ cv2.imshow('Validation', img1)
#~ cv2.waitKey(0)
#~ cv2.destroyAllWindows()

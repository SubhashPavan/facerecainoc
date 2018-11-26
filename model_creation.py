from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
import nn4 as network
import datetime

classifier_filename = os.path.expanduser('./models/lfw_classifier.pkl')


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  
def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

data_dir = 'C:/Users/pavansubhash_t/POC_1-July/Aligned Data/'

dataset = facenet.get_dataset(data_dir)

paths, labels = facenet.get_image_paths_and_labels(dataset)

print('Number of classes: %d' % len(dataset))
print('Number of images: %d' % len(paths))



with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './models')
        print ('MTCNN created')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 50
        image_size = 160
        seed = 42
        pool_type = 'MAX'
        use_lrn = False
        
        print('Loading feature extraction model')
        modeldir = './models'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        tf.train.write_graph(sess.graph_def, './graph', "graph.pb", False)
        tf.summary.FileWriter("logs/", sess.graph)
        print('facenet embedding done !')

        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
        classifier_filename_exp = os.path.expanduser('./models/lfw_classifier.pkl')
        class_names = [ cls.name.replace('_', ' ') for cls in dataset]

        
        print('Training classifier')
        model = SVC(kernel='linear', probability=True)
        classnames = [class_names[x] for x in labels]
        model.fit(emb_array, classnames)
        model_dump = 'GarudAIFaceRecClassifier'+'_'+str(datetime.datetime.now().strftime("%Y%m%d%H%M"))+'.pkl'
        features_dump = 'GarudAIFeatureDump'+'_'+str(datetime.datetime.now().strftime("%Y%m%d%H%M"))+'.pkl'
        
        with open(model_dump, 'wb') as outfile:
            pickle.dump((model), outfile)
            print('Saved classifier model to file "%s"' % model_dump)
            
       
        with open(features_dump, 'wb') as outfile:
            pickle.dump((emb_array,classnames), outfile)
            print('Saved features to file "%s"' % features_dump)



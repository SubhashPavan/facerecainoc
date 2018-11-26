# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 20:31:39 2018

@author: Vaibhav
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import datetime
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
import detect_face
import cv2
import statistics as s
import uuid
import glob
import os

pickle_model_list = glob.glob('C:/Users/pavansubhash_t/POC_1-July/classifier_files/*.pkl')

latest_model_file = max(pickle_model_list, key=os.path.getctime)

pickle_features_list = glob.glob('C:/Users/pavansubhash_t/POC_1-July/features_files/*.pkl')

latest_features_file = max(pickle_features_list, key=os.path.getctime)


def uniqueid():
    unique_id = str(uuid.uuid1())
    return unique_id

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
  
def load_data(img, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    #nrof_samples = len(image_paths)
    #images = np.zeros((nrof_samples, image_size, image_size, 3))
    #for i in range(nrof_samples):
    #img = cv2.imread(image_paths[i])
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    img = crop(img, do_random_crop, image_size)
    img = flip(img, do_random_flip)
        
    #print("shape of img ", img.shape)
    #images[:,:,:] = img
    return img

with open(latest_model_file, 'rb') as infile:
    (model) = pickle.load(infile)

with open(latest_features_file, 'rb') as infile:
    (emb_array,class_names) = pickle.load(infile)


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
        batch_size = None
        image_size = 240
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
        trackerlst = []
        bbox =[]
        oklist= []
        
        video_capture = cv2.VideoCapture(0)
        crops = []
        c = 0

        # #video writer
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(640,480))

        print('Start Recognition!')
        

        while True:
            # Capture frame-by-frame

            ret, frame = video_capture.read()
            uniqueid_init = None
            #cv2.imwrite('Frame.png',frame)
            #print (frame)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(frame.shape)

            #timeF = frame_interval
            try:
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                
                n_faces = bounding_boxes.shape[0]#number of faces
                #print('Number of Faces are：{}'.format(n_faces))
                
                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                
                    
                for face_position in bounding_boxes:
                
                

                    face_position=face_position.astype(int)
                
                    cv2.rectangle(
                      frame,
                      (face_position[0], 
                      face_position[1]), 
                      (face_position[2], face_position[3]),
                      (55,255,155), 2)
                
                    cropped=frame[face_position[1]:face_position[3],face_position[0]:face_position[2],]
    
                    cropped = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC )
                    
                    cropped = load_data(cropped,False,False,160)
                    data=cropped.reshape(-1, 160, 160, 3)
                                    
                    emb_data = sess.run(embeddings, feed_dict={images_placeholder: np.array(data),
                                                     phase_train_placeholder: False })    
                    
                    predict_pb = model.predict_proba(emb_data)
                        
                    best_class_indices = np.argmax(predict_pb, axis=1)
                    best_class_probabilities = predict_pb[np.arange(len(best_class_indices)), best_class_indices]
                                            
                    if best_class_probabilities[0] > 0.9:
                        find_results = str(1)+'-'+str(round(best_class_probabilities[0]*100,1))+'%'+'-'+str(model.classes_[best_class_indices[0]])
                    else:
                        try:
                               count = []
                               distance = []
                               emb_data = emb_data.reshape(512,)

                               for i in range(len(emb_array)):
                                count.append(i)
                                distance.append(np.sqrt(np.sum(np.square(emb_array[i] - emb_data))))
                                    
                                
                               names_dict = dict(zip(count,distance))
                               names_tuple = dict((v,k) for k,v in names_dict.items())
                                
                               sorted_dist = sorted(names_dict.values())
                               sorted_dist = sorted_dist[:5]
                                
                               sorted_names = [names_tuple[x] for x in sorted_dist ]
                                
                               sorted_labels = [class_names[x] for x in sorted_names]
                                
                               if len(sorted_labels)!= len(set(sorted_labels)):
                                    final_result = max(sorted_labels,key = sorted_labels.count)
                                    indices = [i for i, x in enumerate(sorted_labels) if x == final_result]
                                    dist_list = [sorted_dist[x] for x in indices]
                                    final_dis = s.mean(dist_list)
                                    
                               else:
                                    final_result = sorted_labels[0]
                                    final_dis = sorted_dist[0]
                               percentage = min(100,100*0.4/final_dis)

                               if percentage > 90:
                                   find_results = str(2)+'-'+str(round(percentage,2))+'%'+'-'+str(final_result)
                               else:
                                   if uniqueid_init is None:
                                       unique_id = uniqueid()
                                   else:    
                                       emb_data = emb_data.reshape(1,512)
                                       emb_array = np.append(emb_array,emb_data,axis = 0)
                                       class_names = class_names.append(unique_id)
                                       find_results =  str(3)+'-'+str('FD')+'-'+str(unique_id)
                                       features_dump = 'GarudAIFeatureDump'+'_'+str(datetime.datetime.now().strftime("%Y%m%d%H%M"))+'.pkl'
                                       with open(features_dump, 'wb') as outfile:
                                            pickle.dump((emb_array,class_names), outfile)
                                            print('Saved features to file "%s"' % features_dump)
                                            
      
                        except Exception as e:
                            print ("ERROR:",str(e))
                            
                            
                cv2.putText(frame,'{}'.format(find_results),(face_position[0],face_position[1]) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
     
            except Exception as e :
                print ("Error:",str(e))
                pass
                
            
            
            cv2.imshow('GarudAI', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

            


                    





import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

""""Real time streaming object"""
cap = cv2.VideoCapture(0)

"""Downloading a pretrained model"""
download_flag = 0  # =1 if model hasn't been downloaded beforehand
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_EXT = '.tar.gz'
MODEL_IN = MODEL_NAME + MODEL_EXT

if download_flag == 1:
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_IN, MODEL_IN)

"""Extracting the .gz file to obtain the pretrained model in protobuf format"""
#tar_file = tarfile.open(MODEL_IN)
#for file in tar_file.getmembers():
#    file_name = os.path.basename(file.name)
#    if 'frozen_inference_graph.pb' in file_name:
#        tar_file.extract(file, os.getcwd())

PATH_TO_CKPT = os.path.join(os.getcwd(), 'ssd_inception_v2_coco_2018_01_28', 'frozen_inference_graph.pb')

"""Getting the labels list for our detection"""
PATH_TO_LABELS = os.path.join(os.getcwd(), 'object_detection', 'data', 'mscoco_label_map.pbtxt')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

NUM_CLASSES = 90

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

"""Loading the frozen model weights and other params into memory"""
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

"""Object_Detection Code"""
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            """Getting the image(frame in a video) from camera and resizing it"""
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)

            """Getting the parameters from the model for detection"""
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            """Detection"""
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                feed_dict={image_tensor: image_np_expanded})

            """Visualization"""
            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                               category_index, use_normalized_coordinates=True,
                                                               line_thickness=8)

            cv2.imshow('Object Detection', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break







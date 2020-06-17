'''
REFERENCE : DHARUN'S REPO
IMPORTANT OBSERVATIONS
Below script works fine in cropping out the license plate when the resolution of the images are quiet high.
Although it is not able to extract the characters at all using the pytesseract approach.
Some alternate approach should be employed.Try to implement the AI based license plate detector's project code's 
   aproach in oredr to get the characters from the number plates.

'''

import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import cv2
import pytesseract

from custom_plate import allow_needed_values as anv 
from custom_plate import do_image_conversion as dic

sys.path.append("..")

import label_map_util
import visualization_utils as vis_util

MODEL_NAME = 'numplate'
PATH_TO_CKPT = MODEL_NAME + '/graph-200000/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object_detection.pbtxt')
NUM_CLASSES = 1


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'png_tesseract/test_ram'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'i_{}.jpg'.format(i)) for i in range(1, 3) ]
TEST_ANUJ=os.path.join('numplate')
count = 0

'''
Below code from 57-111 is responsible for the object detection part.  
'''

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path) 
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      ymin = boxes[0,0,0]
      xmin = boxes[0,0,1]
      ymax = boxes[0,0,2]
      xmax = boxes[0,0,3]
      (im_width, im_height) = image.size
      (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
      cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn),int(ymaxx - yminn), int(xmaxx - xminn))
      img_data = sess.run(cropped_image)
      
      '''
      below lines 83-88 are trying to extract the characters from the cropped number plate
      i.e the number plate which only has image of the characters.It won't have any car's 
      objects in it.
      '''
      count = 0
      filename = dic.yo_make_the_conversion(img_data, count)
      pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
      text = pytesseract.image_to_string(Image.open(filename),lang="eng") 
      print("Text", text)
      print('CHARACTER RECOGNITION : ',anv.catch_rectify_plate_characters(text))
      
      '''
      below lines 90-97 would put bounding box around the images and label(numplate) the detected number plate
      along with the score.
      '''
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=5)
      
cv2.imshow('image_np',image_np)     
cv2.imshow('img_data',img_data)
cv2.waitKey(0)
cv2.destroyAllWindows()


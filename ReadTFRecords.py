import tensorflow as tf
import cv2
import numpy as np

def AnotherReader(rfRecordFile):
    record_iterator = tf.python_io.tf_record_iterator(path=rfRecordFile)
    num_images = 1
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        # example.features.feature.values() // get info on all avilable values
        height = int(example.features.feature['image/height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['image/width']
                    .int64_list
                    .value[0])
        filename = (example.features.feature['image/filename']
                                  .bytes_list
                                  .value[0])
        xmax = (example.features.feature['image/object/bbox/xmax']
                    .float_list
                    )
        xmin = (example.features.feature['image/object/bbox/xmin']
                .float_list
                )
        ymax = (example.features.feature['image/object/bbox/ymax']
                .float_list
                )
        ymin = (example.features.feature['image/object/bbox/ymin']
                .float_list
                )
        labels = (example.features.feature['image/object/class/label']
                .int64_list)
        labels_text = (example.features.feature['image/object/class/text']
                  .bytes_list)

        image_bytes = (example.features.feature['image/encoded']
                       .bytes_list)
        # image = tf.image.decode_jpeg(example.features.feature['image/encoded'], tf.uint8)
        numberOfBoxes = xmax.value.__len__()

        print('{} file {} height {} widht {} boxes {}   '.format(num_images,filename,height,width,numberOfBoxes))
        num_images = num_images + 1

        #find how to run inferance on the image we output from the tfrecord, and get its boxes and classes.

def _extract_feature(element):
    """
    Extract features from a single example from dataset.
    """
    features = tf.parse_single_example(
        element,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
            'image/height': tf.VarLenFeature(tf.int64),
            'image/object/bbox/xmax':tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmin':tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':tf.VarLenFeature(tf.float32),
            'image/object/class/label':tf.VarLenFeature(tf.int64),
            'image/object/class/text': tf.VarLenFeature(tf.string),
            'image/source_id': tf.VarLenFeature(tf.string),
            'image/width': tf.VarLenFeature(tf.int64)
        })
    return features

def show_record(tfRecord_filename):
    """.
    Show the TFRecord contents
    """
    # Generate dataset from TFRecord file.
    dataset = tf.data.TFRecordDataset(tfRecord_filename)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()

    # Extract features from single example
    features = _extract_feature(next_example)
    image_decoded = tf.image.decode_image(features['image/encoded'])
    label_x = tf.cast(features['image/width'], tf.int64)
    label_y = tf.cast(features['image/height'], tf.int64)
    # image = tf.image.decode_jpeg(features['image/encoded'], tf.uint8)
    # # label = tf.cast(features['image/object/class/label'], tf.int32)
    # height = tf.cast(features['image/height'], tf.int32)
    # width = tf.cast(features['image/width'], tf.int32)


    # Use openCV for preview
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # Actrual session to run the graph.
    with tf.Session() as sess:
        while True:
            try:
                image_tensor, label_text = sess.run(
                    [image_decoded, (label_x, label_y)])

                # Use OpenCV to preview the image.
                image = np.array(image_tensor, np.uint8)
                cv2.imshow("image", image)
                cv2.waitKey(100)
                cv2.imwrite('c:/temp/img1.jpg',image)
                # Show the labels
                print(label_text)
            except tf.errors.OutOfRangeError:
                break

def printAllTfRecordInfoToFile(tfrecord_input_path,outputFilePath):
    f1 = open(outputFilePath, 'w')
    for example in tf.python_io.tf_record_iterator(tfrecord_input_path):
        print(tf.train.Example.FromString(example), file=f1)
    f1.flush()
    f1.close()

# input_path = "C:\\Yolo\\DataSets\\3classes\\Marana\\TFRecords\\onlyIR\\ir_valid.record" #70
input_path = "C:\\Yolo\\DataSets\\3classes\\Marana\\TFRecords\\onlyCCD\\ccd_valid.record" # 127
# input_path = "C:\\Yolo\\DataSets\\3classes\\Marana\\TFRecords\\full\\valid_full.record" # 4538

#read all tfrecord and save contents to file, this we can know what features are stored in the tfrecord file.
# printAllTfRecordInfoToFile(input_path,'c:/temp/ccd_valid.txt')

import sys
print(sys.executable)
# show_record(input_path)

AnotherReader(input_path)

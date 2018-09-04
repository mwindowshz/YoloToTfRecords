import os
from jinja2 import Environment, PackageLoader


class Writer:
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=PackageLoader('pascal_voc_writer', 'templates'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)


import cv2
import os
import csv

def parse_yolo_labels(images_list_file_name=None, VocLabelsDirReplace=True, classes=['background',
                                                                                     'aeroplane', 'bicycle', 'bird',
                                                                                     'boat',
                                                                                     'bottle', 'bus', 'car', 'cat',
                                                                                     'chair', 'cow', 'diningtable',
                                                                                     'dog',
                                                                                     'horse', 'motorbike', 'person',
                                                                                     'pottedplant',
                                                                                     'sheep', 'sofa', 'train',
                                                                                     'tvmonitor'],
                      ret=False):
    '''
    Arguments:
        images_list_file_name (str): images list file containing full path for each file files are the jpg images
        next to each jpg file there should be a txt file containing the yolo_marks
        Yolo marks are as follows, x,y center of rect, width height of rect
        all the values are relative to image size, this is why we need to read the image to get its dimensions

        VOC labels are not in the same folder as the jpeg but in labels folder, so we can read the corresponding
        label from the correct path - this is the use of next argumen:

       VocLabelsDirReplace (bool):  are VOC files jpg and labels in different folders. if True, then replace the
       'JPEGImages','labels', and load the txt file from labels folder
        ret (bool, optional): Whether or not the image filenames and labels are to be returned.
            Defaults to `False`.

    Returns:
        None by default, optionally the image filenames and labels.
    '''

    # Erase data that might have been parsed before

    with open(images_list_file_name) as f:
        # read one image from list, and add its information
        for line in f:
            # open the txt file, so change the ending of file name to be .txt

            imageFile = str(line.rstrip())
            print(imageFile)
            img = cv2.imread(imageFile)
            height, width, c = img.shape
            #create voc writer
            writer = Writer(imageFile, width=width, height=height)

            # self.filenames.append(imageFile)  only add a file if there are labesl
            current_labels = []
            labelFile = imageFile.replace('jpg', 'txt')
            xmlFile = imageFile.replace('jpg', 'xml')
            if VocLabelsDirReplace:
                labelFile = labelFile.replace('JPEGImages', 'labels')

            with open(labelFile) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=' ')
                for row in readCSV:
                    classID = int(row[0]) + 1  # we add 1 because in ssd 0 is background
                    rectHeight = int(float(row[4]) * height)
                    rectWidth = int(float(row[3]) * width)
                    centerX = row[1] * width
                    centerY = row[2] * height
                    xmin = int(float(row[1]) * width) - int(rectWidth / 2)
                    ymin = int(float(row[2]) * height) - int(rectHeight / 2)
                    xmax = xmin + rectWidth
                    ymax = ymin + rectHeight
                    imageName = os.path.split(imageFile)
                    # imageName = ntpath.basename(imageFile)
                    box = []
                    # box.append(imageName)
                    box.append(classID)
                    box.append(xmin)
                    box.append(ymin)
                    box.append(xmax)
                    box.append(ymax)
                    current_labels.append(box)
                    print(box)
                    # data.append(box)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                    writer.addObject(classes[classID], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,difficult=1)
                writer.save(xmlFile)


    # ::save(path)


#
# classes = ['background',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat',
#            'chair', 'cow', 'diningtable', 'dog',
#            'horse', 'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor']

classes = ['background',
            'person', 'animal','vehicle']
#  "D:\\YOLO\\Training\\base\\data\\themalInValid.txt"
trainFiles = "C:/Yolo/DataSets/3classes/Marana/voc/2012/1/all_2012.txt"
parse_yolo_labels(images_list_file_name=trainFiles,classes=classes)
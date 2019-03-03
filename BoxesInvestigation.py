import cv2

import csv
import ntpath
import numpy as np
import os.path
import matplotlib.pyplot as plt


  # 'C:\\Users\\owner\\Source\\Repos\\Yolo_mark\\x64\\Release\\data\\allFiles.txt'
# filenames2,labels2 = train_dataset.parse_yolo_labels(images_list_file_name, True, True)
#
def BoxesStatics(imageList):
    avg_width = 0
    avg_height = 0
    ObjectToImageRatio = 1
    avg_size = 0
    numOfboxes = 0
    minWidth = 1000
    minHeight = 1000
    maxWidth = 0
    maxHeight = 0
    boxesWidth = []
    boxesHeight = []
    boxesSize = []
    boxesToImageRatio = []
    boxSumAreaRatioToImage = []
    labels2 = []
    filenames2 = []
    amoutObjectsForEachClass = [0,0,0] #when we have 3 classes


    with open(images_list_file_name) as f:
        #read one image from list, and add its information
        for line in f:
            # open the txt file, so change the ending of file name to be .txt

            imageFile = str(line.rstrip())
            print(imageFile)

            isfile = os.path.isfile(imageFile)
            if isfile==False:
                continue
            img = cv2.imread(imageFile)
            image_height,image_width,c = img.shape
            filenames2.append(imageFile)
            current_labels = []

            labelFile = imageFile.replace('jpg', 'txt')
            if True:
                labelFile = labelFile.replace('JPEGImages', 'labels')

            with open(labelFile) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=' ')
                boxesAreaInOneImage = 0
                for row in readCSV:
                    classID = int(row[0]) + 1  # we add 1 because in ssd 0 is background
                    rectHeight = int(float(row[4]) * image_height)
                    rectWidth = int(float(row[3]) * image_width)
                    xmin = int(float(row[1]) * image_width) - int(rectWidth / 2)
                    ymin = int(float(row[2]) * image_height) - int(rectHeight / 2)
                    xmax = xmin + rectWidth
                    ymax = ymin + rectHeight
                    imageName = ntpath.basename(imageFile)
                    box = []
                    # box.append(imageName)
                    box.append(classID)
                    box.append(xmin)
                    box.append(ymin)
                    box.append(xmax)
                    box.append(ymax)
                    current_labels.append(box)
                    print(box)
                    boxWidth = xmax - xmin
                    boxheight = ymax - ymin
                    if(boxWidth >0 and boxheight > 0):
                        boxesAreaInOneImage+=boxWidth*boxheight
                        avg_size += boxWidth*boxheight
                        ratio = (boxWidth*boxheight)/(image_width*image_height)
                        ObjectToImageRatio +=  ratio
                        boxesSize.append(boxWidth*boxheight)
                        boxesToImageRatio.append(ratio)
                        avg_width += boxWidth
                        avg_height += boxheight
                        numOfboxes += 1
                        minWidth = min(boxWidth,minWidth)
                        minHeight = min(minHeight,boxheight)
                        maxWidth = max(maxWidth,boxWidth)
                        maxHeight = max(maxHeight,boxheight)
                        boxesWidth.append(boxWidth)
                        boxesHeight.append(boxheight)
                        #amoutObjectsForEachClass[classID] +=1
                    # data.append(box)
                    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)
                    if current_labels.__len__() > 0:  # only add file and labels if there are labels
                        labels2.append((np.stack(current_labels, axis=0)))
            #     go over all boxes in image, an calc their overall coverage against image size
                boxSumAreaRatioToImage.append(boxesAreaInOneImage/(image_width*image_height))
            # cv2.imshow("img",img)
            # cv2.waitKey()
    print("num of boxes {}".format(numOfboxes))
    print("avgWidth {}".format(avg_width/numOfboxes))
    print("avgHeight {}".format(avg_height/numOfboxes))
    print("avg_size {}".format(avg_size/numOfboxes))
    print("avg_Ratio {}".format(ObjectToImageRatio/numOfboxes))

    print("min Width {}".format(minWidth))
    print("min Height {}".format(minHeight))

    print("max Width {}".format(maxWidth))
    print("max Height {}".format(maxHeight))
    print("median width {}".format(np.median(boxesWidth)))

    print("median height {}".format(np.median(boxesHeight)))
    return numOfboxes, boxesSize , boxesToImageRatio , boxSumAreaRatioToImage ,boxesWidth ,boxesHeight , amoutObjectsForEachClass


images_list_file_name = 'C:\\Users\\owner\\Downloads\\Compressed\\VOCtrainval_06-Nov-2007\\all VOC 2007 2012.txt'
                        # 'c:\\Users\\owner\\Downloads\\Compressed\\VOCtrainval_06-Nov-2007\\2007_val.txt'
numOfboxes, boxesSize , boxesToImageRatio , boxSumAreaRatioToImage, boxesWidth ,boxesHeight, amoutObjectsForEachClass = BoxesStatics(images_list_file_name)

                        # 'C:\\Users\\owner\\Pictures\\Images_with_tagging\\Data_Set_2\\allFiles.txt'
images_list_file_name = "C:/Users/owner/Pictures/Images_with_tagging/DataSets/all_ccd.txt"
ccd_numOfboxes, ccd_boxesSize , ccd_boxesToImageRatio  , ccd_boxSumAreaRatioToImage ,ccd_boxesWidth ,ccd_boxesHeight, amoutObjectsForEachClassDay = BoxesStatics(images_list_file_name)
images_list_file_name = "C:/Users/owner/Pictures/Images_with_tagging/DataSets/all_thermal.txt"
therm_numOfboxes, therm_boxesSize , therm_boxesToImageRatio  , thermal_boxSumAreaRatioToImage ,thermal_boxesWidth ,thermal_boxesHeight, amoutObjectsForEachClassIr = BoxesStatics(images_list_file_name)

#box plots
boxplotdata = [boxesToImageRatio,ccd_boxesToImageRatio,therm_boxesToImageRatio]
sumBoxPlot = [boxSumAreaRatioToImage,ccd_boxSumAreaRatioToImage,thermal_boxSumAreaRatioToImage]
l = ['VOC','ccd','therm']
plt.boxplot(boxplotdata,labels=l)
plt.title('boxes to image ratio')
plt.show()

plt.boxplot(sumBoxPlot,labels=l)
plt.title('total boxes area to image ratio')
plt.show()


plt.hist(boxesToImageRatio,alpha=0.5,bins='auto',normed=True,label='VOC')
plt.hist(ccd_boxesToImageRatio,alpha=0.5,bins='auto',normed=True,label='CCD')
plt.hist(therm_boxesToImageRatio,alpha=0.5,bins='auto',normed=True,label='Therm')
plt.title("normed")
plt.legend(loc='upper right')
plt.show()

plt.hist(boxesToImageRatio,alpha=0.5,bins='auto',normed=False,label='VOC')
plt.hist(ccd_boxesToImageRatio,alpha=0.5,bins='auto',normed=False,label='CCD')
plt.hist(therm_boxesToImageRatio,alpha=0.5,bins='auto',normed=False,label='Therm')
plt.title("no norm")
plt.legend(loc='upper right')
plt.show()

plt.hist(boxesToImageRatio,alpha=0.5,bins='auto', histtype='stepfilled',normed=False,label='VOC')
plt.hist(ccd_boxesToImageRatio,alpha=0.5,bins='auto', histtype='stepfilled',normed=False,label='CCD')
plt.hist(therm_boxesToImageRatio,alpha=0.5,bins='auto', histtype='stepfilled',normed=False,label='Therm')
plt.title("same")
plt.legend(loc='upper right')
plt.show()


plt.hist(boxesToImageRatio,alpha=0.5,bins=20,normed=True,label='VOC')
plt.hist(ccd_boxesToImageRatio,alpha=0.5,bins=20,normed=True,label='CCD')
plt.hist(therm_boxesToImageRatio,alpha=0.5,bins=20, normed=True,label='Therm')
plt.title("same - bins 20 normed")
plt.legend(loc='upper right')
plt.show()

plt.hist(boxesToImageRatio,alpha=0.5,bins=30,normed=False,label='VOC')
plt.hist(ccd_boxesToImageRatio,alpha=0.5,bins=30,normed=False,label='CCD')
plt.hist(therm_boxesToImageRatio,alpha=0.5,bins=30, normed=False,label='Therm')
plt.title("same - bins 30 not normed")
plt.legend(loc='upper right')
plt.show()


n, bins, patches = plt.hist(boxesToImageRatio,alpha=0.5,bins='auto',normed=True,label='VOC')
plt.hist(ccd_boxesToImageRatio,alpha=0.5,bins=bins,normed=True,label='CCD')
plt.hist(therm_boxesToImageRatio,alpha=0.5,bins=bins, normed=True,label='Therm')

plt.title("bins size the same Normed")
plt.legend(loc='upper right')
plt.show()

n, bins, patches = plt.hist(boxesSize,alpha=0.5,bins='auto',normed=True,label='VOC')
plt.hist(ccd_boxesSize,alpha=0.5,bins=bins,normed=True,label='CCD')
plt.hist(therm_boxesSize,alpha=0.5,bins=bins, normed=True,label='Therm')

plt.title("boexes size - Normed -same bins")
plt.legend(loc='upper right')
plt.show()

print("VOC avg {} median {} max {} min {}".format(np.average(boxesToImageRatio),np.median(boxesToImageRatio),np.max(boxesToImageRatio),np.min(boxesToImageRatio)))
print("CCD avg {} median {} max {} min {}".format(np.average(ccd_boxesToImageRatio),np.median(ccd_boxesToImageRatio),np.max(ccd_boxesToImageRatio),np.min(ccd_boxesToImageRatio)))
print("Thermal avg {} median {} max {} min {}".format(np.average(therm_boxesToImageRatio),np.median(therm_boxesToImageRatio),np.max(therm_boxesToImageRatio),np.min(therm_boxesToImageRatio)))
#
# h = plt.hist(boxesToImageRatio,bins=20,color='b',alpha=0.3,label='theoretical',histtype='stepfilled', normed=True)
# p = h[2][0]
# p.xy[:,1] /= p.xy[:, 1].max()
# h = plt.hist(ccd_boxesToImageRatio,bins=20,alpha=0.5,color='g',label='experimental',histtype='stepfilled',normed=True)
# p = h[2][0]
# p.xy[:,1] /= p.xy[:, 1].max()

from scipy.stats import kurtosis
kurtosis(boxesToImageRatio)
from scipy import stats


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

mu = 0
variance = 1
sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x,mlab.normpdf(x, mu, sigma))
# plt.show()

plt.plot(np.array(ccd_boxesToImageRatio),mlab.normpdf(np.array(ccd_boxesToImageRatio),mu,sigma))
plt.show()

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def xml_to_csv_from_list(images_list_file_name):
    xml_list = []
    with open(images_list_file_name) as f:
      for line in f:


        imageFile = str(line.rstrip())
        xml_file = imageFile.replace('jpg', 'xml')

        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
          value = (imageFile,#root.find('filename').text,
                       int(root.find('size')[0].text),
                       int(root.find('size')[1].text),
                       member[0].text,
                       int(member[4][0].text),
                       int(member[4][1].text),
                       int(member[4][2].text),
                       int(member[4][3].text)
                       )

          xml_list.append(value)
      column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
      xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
  source_file_list = "C:\\Yolo\\DataSets\\3classes\\\ir_train.txt"  # CHANGE PATH TO YOUR FILE - LIST of jpg files with xml file next to them in the same directory
  dest_csv_file = 'C:\\Yolo\\DataSets\\3classes\\CSV_list_File\\ir_train.csv'  # CHANGE PATH TO YOUR FILE
  df = xml_to_csv_from_list(source_file_list)
  df.to_csv(dest_csv_file,index=None)
  # image_path = os.path.join(os.getcwd(), 'annotations')
  # xml_df = xml_to_csv(image_path)
  # xml_df.to_csv('raccoon_labels.csv', index=None)
  # print('Successfully converted xml to csv.')


main()
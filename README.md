# YoloToTfRecords
tools for converting yolo boxes to pascal voc xml and TFRecords
The repository contain tools for converting from YOLO boxes format to PASCAL VOC xml format and to TFRecords format

in DATA folder are examples of YOLO txt boxes format, and PASCAL VOC xml format.

Steps to create TFREcords
1. convert YOLO txt to PASCAL VOC xml format using provided tools
2. Create CSV list file using xml_to_csv 
  fill in 
   source_file_list = "C:\\Yolo\\DataSets\\3classes\\ir_train.txt"
  dest_csv_file = 'C:\\Yolo\\DataSets\\3classes\\CSV_list_File\\ir_train.csv'
  
  with input list of jpg files with full path (next to each jpg file should be the xml boxes file), output name of csv file.
3. Create TFrecords from CSV file:
  After created csv file, run the following:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  
  where you should set the csv_input as full path to your csv file, and output_path full path and file name of .record file.
  use / slashe, 
  
  Example of usage:
   python generate_tfrecord.py --csv_input='C:/Yolo/DataSets/3classes/CSV_list_File/ir_train.csv' --output_path = C:/Yolo/DataSets/3classes/train.records
   

# YoloToTfRecords
tools for converting yolo boxes to pascal voc xml and TFRecords
The repository contain tools for converting from YOLO boxes format to PASCAL VOC xml format and to TFRecords format

in DATA folder are examples of YOLO txt boxes format, and PASCAL VOC xml format.

Steps to create TFREcords
1. convert YOLO txt to PASCAL VOC xml format using provided tools

  1.1 Enter PascalVocWriter Folder

  1.2 open __init__.py, on the end you would find the code, to set calss names, pas a list of all jpg images wich hava a txt with YOLO Marks format file next to them

    
    > classes = ['background',
                'person', 'animal','vehicle']
    
    > trainFiles = "D:\\YOLO\\img-s11\\img-s1\\img\\all.txt"
    
    > parse_yolo_labels(images_list_file_name=trainFiles,classes=classes)
    
 run code the result should be an additional xml file for each image


2. Create CSV list file using xml_to_csv 
  fill in 
   source_file_list = "C:\\Yolo\\DataSets\\3classes\\ir_train.txt"
  dest_csv_file = 'C:\\Yolo\\DataSets\\3classes\\CSV_list_File\\ir_train.csv'
  
  with input list of jpg files with full path (next to each jpg file should be the xml boxes file), output name of csv file.
  
  
      ### CSV File Format
     > filename,width,height,class,xmin,ymin,xmax,ymax

    > D:\YOLO\img-s11\img-s1\Images\voc\2007\1\000001.jpg,353,500,animal,48,240,195,371

3. Create TFrecords from CSV file:
  After created csv file, run the following:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  
  where you should set the csv_input as full path to your csv file, and output_path full path and file name of .record file.
  use / slashe, 
  
  Example of usage:
   python generate_tfrecord.py --csv_input='C:/Yolo/DataSets/3classes/CSV_list_File/ir_train.csv' --output_path = C:/Yolo/DataSets/3classes/train.records
   
 # 4. use TensorFlow Detection API for training and evaluation 
 see also https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73
  ## 4.1 download and install TF Detection API
  https://github.com/tensorflow/models/tree/master/research/object_detection
  follow the instructions on readme.md
  see also tutorial in https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/
  
  steps to install object detection API
  
  download git repo
  install
  pip install pillow
  
  pip install lxml
  
  pip install jupyter
  
  pip install matplotlib
  
  Head to the protoc releases page and download the protoc-3.4.0-win32.zip, extract it, and you will find protoc.exe in the bin directory.
  https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip
  
You can move this to something more appropriate if you like, or leave it here. I eventually put mine in program files, making a "protoc" directory and dropping it in there.
Now, from within the models (or models-master) directory, you can use the protoc command like so:

"C:/Program Files/protoc/bin/protoc" object_detection/protos/*.proto --python_out=.

#### and very important from research folder call:
  set PYTHONPATH=%PYTHONPATH%;%cd%;%cd%\slim
  
  ## 4.2 choose model to use for transfer learning
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  
  ## 4.3 
   edit model config file:
   See step 3 in https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73
   
   set number of classes
   path for tfrecords train.record. and eval.record
   
   then run training Step 4: Evaluating the model
   python train.py --logtostderr \ 
       --train_dir=training/ \       
 --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
 
 
 
 set PYTHONPATH=%PYTHONPATH%;%cd%;%cd%\slim
cd C:\Yolo\DataSets\Data_For_TF_Obecect_detection_API\models-master\research\object_detection
python .\legacy\train.py --logtostderr --train_dir=D:/tf-od-api/first/training/ --pipline_config_path=D:/tf-od-api/first/training/ssd_mobilenet_v1_coco.config

python .\legacy\eval.py    --logtostderr --pipeline_config_path=D:/tf-od-api/first/training/ssd_mobilenet_v1_coco.config   --checkpoint_dir=D:/tf-od-api/first/training/ --eval_dir=D:/tf-od-api/first/eval/

#To visualize the eval results
tensorboard --logdir=D:/tf-od-api/first/eval/

#TO visualize the training results
tensorboard --logdir=D:/tf-od-api/first/training/
 ### Use BatchRunner.py to run set of training and evaluation
 
 BatchRunner.py is a script using object detection api
 Place BatchRunner.py in models-master\research\object_detection folder.
 
 prepare a Models folder, and a training/eval folder
 
 set the following variables:
   modelBaseDir = 'D:\\tf-od-api\\3classes\\Base_Modeles_Dir'
  trainBaseDir = 'D:\\tf-od-api\\3classes\\Train_Eval_Base_Dir'
  trainPyFilePath = '.\\legacy\\train.py'
  evalPyFilePath = '.\\legacy\\eval.py'
  
  after this run the BatchRunner.py file
  Python BatchRunner.py
  
  this would loop over each model prepared in modelBaseDir directory
  and run training and eval.
  Set trainBaseDir directory to be used.
  
 
   
   

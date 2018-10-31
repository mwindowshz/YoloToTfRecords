import os
import tensorflow as tf
import pandas as pd

def printTrainingLossResults(train_dir,fileToWriteResults):
    finalResults = []
    for file in os.listdir(train_dir):
        if file.startswith("events.out.tfevents"):
            print(file)
            s = train_dir + '/' + file
            print(s)
            a = tf.train.summary_iterator(s)
            for Event in a:
                for value in Event.summary.value:
                    if 'classification_loss' in value.tag:
                        finalResults.append(value)
                    if 'localization_loss' in value.tag:
                        finalResults.append(value)
                    # print(value)
    print("\n")
    # print(finalResults)
    for item in finalResults:
        print(item, file=fileToWriteResults, flush=True)
    fileToWriteResults.flush()

def printEvalAPResults(eval_dir,fileToWriteResults):
    evalValus = []
    for file in os.listdir(eval_dir):
        if file.startswith("events.out.tfevents"):
            print(file)
            s = eval_dir + '/' + file
            print(s)
            a = tf.train.summary_iterator(s)
            for Event in a:
                for value in Event.summary.value:
                    if 'PascalBoxes' in value.tag:
                        evalValus.append(value)
    #write the data to the file
    retValues = []
    for item in evalValus:
        print(item, file=fileToWriteResults, flush=True)
        retValues.append(item.simple_value)
    fileToWriteResults.flush()

    print(evalValus)
    return retValues

#**************vaiables to set by user:******************
#
modelBaseDir = 'D:\\tf-od-api\\3classes\\Base_Modeles_Dir'
trainBaseDir = 'D:\\tf-od-api\\3classes\\Train_Eval_Base_Dir_200000'

# modelBaseDir = 'D:/tf-od-api/3classes/testBase'
# trainBaseDir = 'D:\\tf-od-api\\3classes\\test_Train_Eval'

trainPyFilePath = '.\\legacy\\train.py'
evalPyFilePath = '.\\legacy\\eval.py'
DO_TRAINING = False
DO_EVAL = True
doFullEval = True
OnlyExtractResults = True # for only extracting results, not doing the acctual eval - used
#**************END vaiables to set by user:******************

#eval result index
index = ['PascalBoxes_Precision/mAP@0.5IOU/animal',
         'PascalBoxes_Precision/mAP@0.5IOU/person',
         'PascalBoxes_Precision/mAP@0.5IOU/vehicle',
         'PascalBoxes_Precision/mAP@0.5IOU']


if not os.path.exists(trainBaseDir):
    os.makedirs(trainBaseDir)
#eval configs are diffrent for each group of images we want to eval,

evalNames = ['pipeline_ir', 'pipeline_ccd', 'pipeline_full', 'pipeline_voc']
pipelineEvalConfigs = []
#list of all model directorys

modelsDirList = [f.path for f in os.scandir(modelBaseDir ) if f.is_dir() ]
#open log file and clear it
fname = trainBaseDir + "/" + "results.txt"
fileToWriteResults = open(fname, "w")
fileToWriteResults.close()

#loop over all models in model list and run train then eval on the training.
for modelPath in modelsDirList:
    pipeline_config_path = modelPath + '/' + 'pipeline.config'
    modelName = modelPath.rsplit('\\', 1)[-1]
    train_dir = trainBaseDir + '/' + modelName + '/' + 'train'
    eval_dir = trainBaseDir + '/' + modelName + '/' + 'eval'
    #create directory for eval and train,
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    print('pipeline_config_path : ' + pipeline_config_path + '\n')
    print('modelName : ' + modelName + '\n')
    print('train_dir : ' + train_dir + '\n')
    print('eval_dir : ' + eval_dir + '\n')

    trainCommand = 'python ' + trainPyFilePath + ' --logtostderr --train_dir=' + train_dir + ' --pipeline_config_path=' + pipeline_config_path
    print(trainCommand)
    print('\n')
    evalCommand = 'python ' + evalPyFilePath + ' --logtostderr --pipeline_config_path=' + pipeline_config_path +  ' --checkpoint_dir=' + train_dir + ' --eval_dir=' + eval_dir
    print(evalCommand)
    print('\n')
    print('\n')
    print('\n')
    fname = trainBaseDir + "/" + modelName + "results.txt"
    fileToWriteResults = open(fname, "a+")
    if DO_TRAINING:
        '''RUN TRAINING ON EACH MODEL'''
        fileToWriteResults.write(trainCommand)
        os.system(trainCommand)
        print('^^^^^ finished training^^^ results are \n')
        printTrainingLossResults(train_dir,fileToWriteResults)
    #add time mesurments to know how long this took
    if DO_EVAL:
        if doFullEval:
            #create dataframe for csv results output
            df = pd.DataFrame(index=index)
            # create eval command and run it
            for name in evalNames:
                path, ending = pipeline_config_path.split('pipeline')
                tempEvalConfig = path + name + ending
                curr_eval_dir = eval_dir+name
                if not os.path.exists(curr_eval_dir):
                    os.makedirs(curr_eval_dir)
                evalCommand = 'python ' + evalPyFilePath + ' --logtostderr --pipeline_config_path=' + tempEvalConfig + ' --checkpoint_dir=' + train_dir + ' --eval_dir=' + curr_eval_dir
                print(evalCommand)
                fileToWriteResults.write('\n{}\n {}'.format(name,evalCommand))
                if not OnlyExtractResults:
                    os.system(evalCommand)
                    print('finished Eval {}'.format(name))
                evalValues = printEvalAPResults(curr_eval_dir, fileToWriteResults)
                if len(evalValues) > 0:
                    df[name] = pd.Series(index=index, data=evalValues)

            csv_file_name = trainBaseDir + "/" + modelName + "results.csv"
            df.to_csv(csv_file_name)

        else:
            fileToWriteResults.write(evalCommand)
            os.system(evalCommand)
            print('^^^ finished eval ^^^')
            printEvalAPResults(eval_dir,fileToWriteResults)
    #
    fileToWriteResults.close()

# os.system('python .\\legacy\\train.py --logtostderr --train_dir=D:/tf-od-api/3classes/3/traindir --pipeline_config_path=D:/tf-od-api/3classes/3/ssd_mobilenet_v1_coco_11_06_2017/pipeline.config')
# print("Finished training")
# finalResults = []
# for file in os.listdir(sTraingDir):
#     if file.startswith("events.out.tfevents"):
#         print(file)
#         s = sTraingDir + '/' +  file
#         print(s)
#         a = tf.train.summary_iterator(s)
#         for Event in a:
#             for value in Event.summary.value:
#                 if 'classification_loss' in value.tag:
#                     finalResults.append(value)
#                 if 'localization_loss' in value.tag:
#                     finalResults.append(value)
#                 # print(value)
# print("\n")
# print(finalResults)
#
#
#
# # os.system('python .\\legacy\\eval.py    --logtostderr --pipeline_config_path=D:/tf-od-api/3classes/3/ssd_mobilenet_v1_coco_11_06_2017/pipeline.config   --checkpoint_dir=D:/tf-od-api/3classes/3/traindir  --eval_dir=D:/tf-od-api/3classes/3/eval')
# print("Finished evaluation")
# evalValus = []
# for file in os.listdir(sEvalDir):
#     if file.startswith("events.out.tfevents"):
#         print(file)
#         s = sEvalDir + '/' + file
#         print(s)
#         a = tf.train.summary_iterator(s)
#         for Event in a:
#             for value in Event.summary.value:
#                 if 'PascalBoxes' in value.tag:
#                     evalValus.append(value)
#
# print(evalValus)
# # a = tf.train.summary_iterator('D:/tf-od-api/3classes/3/eval')
#
# index = ['PascalBoxes_Precision/mAP@0.5IOU',
#          'PascalBoxes_Precision/mAP@0.5IOU/animal',
#          'PascalBoxes_Precision/mAP@0.5IOU/person',
#          'PascalBoxes_Precision/mAP@0.5IOU/vehicle']
#
# trainBaseDir = 'c:/temp'
# modelName = 'ssd_v1'
#
# if True:
#     df = pd.DataFrame(index = index,columns=[modelName])
#     df[modelName] = pd.Series(index=index,data=evalValues)
#     print(df)
#     print('loop another model\n')
#     modelName = 'ssd_v2'
#     evalValues = [0.02,0.5,0.6,0.01]
#     df[modelName] = pd.Series(index=index,data=evalValues)
#     print(df)

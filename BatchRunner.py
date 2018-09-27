import os
import tensorflow as tf


def printTrainingLossResults(train_dir):
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
    print(finalResults)
def printEvalAPResults(eval_dir):
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

    print(evalValus)

#**************vaiables to set by user:******************

modelBaseDir = 'D:\\tf-od-api\\3classes\\Base_Modeles_Dir'
trainBaseDir = 'D:\\tf-od-api\\3classes\\Train_Eval_Base_Dir'
trainPyFilePath = '.\\legacy\\train.py'
evalPyFilePath = '.\\legacy\\eval.py'

#list of all model directorys

modelsDirList = [f.path for f in os.scandir(modelBaseDir ) if f.is_dir() ]

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
    evalCommand = 'python ' + evalPyFilePath + ' --logtostderr --pipeline_config_path=' + pipeline_config_path +  ' --checkpoint_dir=' + train_dir + ' --eval_dir=' + eval_dir
    print(evalCommand)

    # printTrainingLossResults(train_dir)


    os.system(trainCommand)
    print('^^^^^ finished training^^^ results are \n')
    printTrainingLossResults(train_dir)
    #add time mesurments to know how long this took
    os.system(evalCommand)
    print('^^^ finished eval ^^^')
    printEvalAPResults(eval_dir)

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

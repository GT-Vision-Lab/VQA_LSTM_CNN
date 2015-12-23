"""
Download the vqa data and preprocessing.

Version: 1.0
Contributor: Jiasen Lu
"""


# Download the VQA Questions from http://www.visualqa.org/download.html
import json
import os
import argparse

def download_vqa():
    os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip -P zip/')

    # Download the VQA Annotations
    os.system('wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip -P zip/')


    # Unzip the annotations
    os.system('unzip zip/Questions_Train_mscoco.zip -d annotations/')
    os.system('unzip zip/Questions_Val_mscoco.zip -d annotations/')
    os.system('unzip zip/Questions_Test_mscoco.zip -d annotations/')
    os.system('unzip zip/Annotations_Train_mscoco.zip -d annotations/')
    os.system('unzip zip/Annotations_Val_mscoco.zip -d annotations/')


def main(params):
    if params['download'] == 'True':
        download_vqa()

    '''
    Put the VQA data into single json file, where [[Question_id, Image_id, Question, multipleChoice_answer, Answer] ... ]
    '''

    train = []
    test = []
    imdir='%s/COCO_%s_%012d.jpg'

    if params['split'] == 1:

        print 'Loading annotations and questions...'
        train_anno = json.load(open('annotations/mscoco_train2014_annotations.json', 'r'))
        val_anno = json.load(open('annotations/mscoco_val2014_annotations.json', 'r'))

        train_ques = json.load(open('annotations/MultipleChoice_mscoco_train2014_questions.json', 'r'))
        val_ques = json.load(open('annotations/MultipleChoice_mscoco_val2014_questions.json', 'r'))

        subtype = 'train2014'
        for i in range(len(train_anno['annotations'])):
            ans = train_anno['annotations'][i]['multiple_choice_answer']
            question_id = train_anno['annotations'][i]['question_id']
            image_path = imdir%(subtype, subtype, train_anno['annotations'][i]['image_id'])

            question = train_ques['questions'][i]['question']
            mc_ans = train_ques['questions'][i]['multiple_choices']

            train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
        
        subtype = 'val2014'
        for i in range(len(val_anno['annotations'])):
            ans = val_anno['annotations'][i]['multiple_choice_answer']
            question_id = val_anno['annotations'][i]['question_id']
            image_path = imdir%(subtype, subtype, val_anno['annotations'][i]['image_id'])

            question = val_ques['questions'][i]['question']
            mc_ans = val_ques['questions'][i]['multiple_choices']

            test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans})
    else:
        print 'Loading annotations and questions...'
        train_anno = json.load(open('annotations/mscoco_train2014_annotations.json', 'r'))
        val_anno = json.load(open('annotations/mscoco_val2014_annotations.json', 'r'))

        train_ques = json.load(open('annotations/MultipleChoice_mscoco_train2014_questions.json', 'r'))
        val_ques = json.load(open('annotations/MultipleChoice_mscoco_val2014_questions.json', 'r'))
        test_ques = json.load(open('annotations/MultipleChoice_mscoco_test2015_questions.json', 'r'))
        
        subtype = 'train2014'
        for i in range(len(train_anno['annotations'])):
            ans = train_anno['annotations'][i]['multiple_choice_answer']
            question_id = train_anno['annotations'][i]['question_id']
            image_path = imdir%(subtype, subtype, train_anno['annotations'][i]['image_id'])

            question = train_ques['questions'][i]['question']
            mc_ans = train_ques['questions'][i]['multiple_choices']

            train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})

        subtype = 'val2014'
        for i in range(len(val_anno['annotations'])):
            ans = val_anno['annotations'][i]['multiple_choice_answer']
            question_id = val_anno['annotations'][i]['question_id']
            image_path = imdir%(subtype, subtype, val_anno['annotations'][i]['image_id'])

            question = val_ques['questions'][i]['question']
            mc_ans = val_ques['questions'][i]['multiple_choices']

            train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
        
        subtype = 'test2015'
        for i in range(len(test_ques['questions'])):
            question_id = test_ques['questions'][i]['question_id']
            image_path = imdir%(subtype, subtype, test_ques['questions'][i]['image_id'])

            question = test_ques['questions'][i]['question']
            mc_ans = test_ques['questions'][i]['multiple_choices']

            test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans})

    print 'Training sample %d, Testing sample %d...' %(len(train), len(test))

    json.dump(train, open('vqa_raw_train.json', 'w'))
    json.dump(test, open('vqa_raw_test.json', 'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--download', default='False', help='Download and extract data from VQA server')
    parser.add_argument('--split', default=1, type=int, help='1: train on Train and test on Val, 2: train on Train+Val and test on Test')
  
    args = parser.parse_args()
    params = vars(args)
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)










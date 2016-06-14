# Deeper LSTM+ normalized CNN for Visual Question Answering

Train a deeper LSTM and normalized CNN Visual Question Answering model. This current code can get **58.16** on Open-Ended and **63.09** on Multiple-Choice on **test-standard** split. You can check [Codalab leaderboard](https://competitions.codalab.org/competitions/6961#results) for more details.

**New VQA Model with better performance and cleaner code can be found here [https://github.com/jiasenlu/HieCoAttenVQA](https://github.com/jiasenlu/HieCoAttenVQA)**

### Requirements

This code is written in Lua and requires [Torch](http://torch.ch/). The preprocssinng code is in Python, and you need to install [NLTK](http://www.nltk.org/) if you want to use NLTK to tokenize the question.

### Ubuntu installation

Installation on other platforms is likely similar, although the command won't be `apt-get`, and the names of the system packages may be different.

```bash
sudo apt-get install libpng-dev libtiffio-dev libhdf5-dev
pip install pillow
pip install -r requirements.txt
python -c "import nltk; nltk.download('all')"
```

### Troubleshooting the prepro.py installation

If when running `prepro.py` you get the error:

```AttributeError: 'module' object has no attribute 'imread'```

Then this means that you're missing the Python library `pillow`. Confusingly, `scipy` rewrites its `scipy.misc` module
depending on what modules are available when the library was installed. To fix it, do:

```bash
pip uninstall scipy
pip install pillow
pip install --no-cache-dir scipy
```

### Evaluation

We have prepared everything for you. 

If you want to train on **train set** and evaluate on **validation set**, you can download the features from [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/data_train_val.zip), and the pretrained model from [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/pretrained_lstm_train_val.t7.zip), put it under the main folder, run 

```
$ th eval.lua -input_img_h5 data_img.h5 -input_ques_h5 data_prepro.h5 -input_json data_prepro.json -model_path model/lstm.t7
```

This will generate the answer json file both on Open-Ended and Multiple-Choice. To evaluate the accuracy of generate result, you need to download the [VQA evaluation tools](https://github.com/VT-vision-lab/VQA).

If you want to train on **train + validation set** and evaluate on **test set**, you can download the feature from [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_val/data_train-val_test.zip), and the pretrained model from [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_val/pretrained_lstm_train-val_test). The rest is the same as previous one, except you need to evaluate the generated results on [testing server](http://www.visualqa.org/challenge.html).

### Training

The first thing you need to do is to download the data and do some preprocessing. Head over to the `data/` folder and run 

```
$ python vqa_preprocessing.py --download True --split 1
```

`--download Ture` means you choose to download the VQA data from the [VQA website](http://www.visualqa.org/) and `--split 1` means you use COCO train set to train and validation set to evaluation. `--split 2 ` means you use COCO train+val set to train and test set to evaluate. After this step, it will generate two files under the `data` folder. `vqa_raw_train.json` and `vqa_raw_test.json`

Once you have these, we are ready to get the question and image features. Back to the main folder, run

```
$ python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 1000
``` 

to get the question features. `--num_ans` specifiy how many top answers you want to use during training. You will also see some question and answer statistics in the terminal output. This will generate two files in your main folder, `data_prepro.h5` and `data_prepro.json`. To get the image features, run

```
$ th prepro_img.lua -input_json data_prepro.json -image_root path_to_image_root -cnn_proto path_to_cnn_prototxt -cnn_model path to cnn_model
```

Here we use VGG_ILSVRC_19_layers [model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77). After this step, you can get the image feature `data_img.h5`. We have prepared everything and ready to launch training. You can simply run

```
$ th train.lua
``` 

with the default parameter, this will take several hours on a sinlge Tesla k40 GPU, and will generate the model under `model/save/`

### Reference

If you use this code as part of any published research, please acknowledge the following repo
```
@misc{Lu2015,
author = {Jiasen Lu and Xiao Lin and Dhruv Batra and Devi Parikh},
title = {Deeper LSTM and normalized CNN Visual Question Answering model},
year = {2015},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/VT-vision-lab/VQA_LSTM_CNN}},
commit = {6c91cb9}
}
```
If you use the VQA dataset as part of any published research, please acknowledge the following paper
```
@InProceedings{Antol_2015_ICCV,
author = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
title = {VQA: Visual Question Answering},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {December},
year = {2015}
}
```
### License

BSD License.

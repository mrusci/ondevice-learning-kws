# Few-Shot Open-Set Learning for On-Device Customization of KWS


This repository contains the code to reproduce the experiments of our [paper](https://arxiv.org/pdf/2306.02161.pdf). 
We provide the scripts to:
* train a fearure extractor using the Multilingual Spoken Words Corpus ([MSWC](https://mlcommons.org/en/multilingual-spoken-words/)) dataset with different training recipes and backbones (refer to the [training details](#feature-extractor-training) for more info)
* run tests on the Google Speech Commands ([GSC](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)) dataset using few-shot examples to initilize a classifier that is meant to work in open-set

Please cite our paper in case you make reuse any part of the code:
```
@inproceedings{rusci_interspeech23,
  author={Manuele Rusci and Tinne Tuytelaars},
  title={{Few-Shot Open-Set Learning for On-Device Customization of KeyWord Spotting Systems}},
  year=2023,
  booktitle={Proc. Interspeech},
}
```
## Repository Overview
All the used scripts are included in the `KWSFSL/` folder:
* `metric_learning.py`: main training script
* `test_fewshots_classifiers_openset.py`: main test script, including the metric calculation
* `data/`: scripts to load MSWC and GSC data
* `classfiers/`: classfiers used for the few-shot inizialization
* `models/`: collection of loss functions and backbones exprimented


## Data Preparation
After defining a`<dataset_path>`, follow the following instrcution to setup the MSWC and GSC datasets.
Also note that additive noise from the DEMAND dataset is used at training time. 

### Multilingual Spoken Words Corpus (MSWC) 
- Simply [download](https://mlcommons.org/en/multilingual-spoken-words/) and unpack the engish partition inside the `<dataset_path>`. Audio files will be in `<dataset_path>/MSWC/en/clips/`
- Convert the audio files to .opus to .wav and store to the outputs to `<dataset_path>/MSWC/en/clips_wav/`. This will fasten the file loads at runtime (no uncompress is needed) at the cost of a higher memory storage. If this step is not done, modify the folder name at line 390 of the `MSWCData.py` file
- Put the split csv files (`en_{train,test,dev}.csv`) to the `<dataset_path>/MSWC/en/` folder
- Add the noise folder to sample the noise recordings: `<dataset_path>/MSWC/noise/`. We used samples from the [DEMAND](https://zenodo.org/record/1227121) dataset, only copying the wav file with ID=01 of every noise type to the destination folder (the name of the file is the destination folder can be any).

### Google Speech Commands (GSC)
The Google Speech Command dataset v2 is unpacked to `<dataset_path>/GSC/`. 
Any link for download can be used (e.g. [torchaudio](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html)).


## Open-Set Test Framework with Few-Shot Example enrollements
After [training](#feature-extractor-training), the feature extractor is evaluated on a few-shot open-set problem. _E.g._: 
```
python KWSFSL/test_fewshots_classifiers_openset.py --data.cuda --speech.dataset googlespeechcommand --speech.task GSC12,GSC22 --speech.include_unknown --fsl.test.n_way 11 --fsl.test.n_episodes 10 --speech.default_datadir <dataset_path>/GSC/ --fsl.test.batch_size 264  --fsl.classifier ncm --fsl.test.n_support 10 --model.model_path results/TL_MSWC500U_DSCNNLLN/best_model.pt
```
As an example, we provide a DSCNNL_NORM model trained using the triplet loss on the MSWC dataset. The checkpoint file can be found in 'results/TL_MSWC500U_DSCNNLLN'.

### Main Test Options
- `fsl.classifier`. Type of the classifier. Options: ['ncm', 'ncm_openmax', 'dproto']. For _dproto_, the feature extractor has to be trained accordingly (dproto loss).
- `fsl.test.n_support`. Number of support samples used to initialize the classifier. Options: []
- `fsl.test.n_way`. Number of classes: N + 1 (_unknown_).
- `fsl.test.n_episodes`. Number of test episodes. At every episode, a different set of support samples is loaded.
- `model.model_path`. Path to the trained model (_pt_ file).


## Feature Extractor Training
To train the feature encoder on the MSWC dataset you can run:
```
python KWSFSL/metric_learning.py --data.cuda --speech.dataset MSWC  --speech.task MSWC500U --speech.default_datadir <dataset_path>/MSWC_en/en/ --speech.include_noise --model.model_name repr_conv --model.encoding DSCNNL_LAYERNORM --model.z_norm   --train.epochs 40  --train.n_way 80 --train.n_support 0 --train.n_query 20 --train.n_episodes 400 --train.loss triplet  --train.margin 0.5  --log.exp_dir <TEST_NAME>/
```
Make sure to set: `dataset_path` and `TEST_NAME`. 
You can use the command above to generate the trained model provided as an example in 'results/TL_MSWC500U_DSCNNLLN'. 

### Main Training Options
- `speech.dataset`. Name of the dataset: 'MSWC'
- `speech.task`. 'MSWC500U' means 500 classes with an Unbalanced number of samples.
- `speech.include_noise`. If enabled, additive noise is added to the utterance samples.
- `model.model_name`. 'repr_conv' indicates the class of the model.
- `model.encoding`. Used encoder. DSCNNL_LAYERNORM is a DSCNN large model with a final layer norm layer. Check models/repr_model.py for more options.
- `model.z_norm`. If enabled, L2 normalization is applied on the embeddings. 
- `train.n_way`. Number of classes for training episodes.
- `train.n_support`. Number of support samples per training episode. Must be different from zero only if using prototypical loss.
- `train.n_query`. Number of samples per training episodes.
- `train.n_episodes`. Number of episodes for epoch.
- `train.loss`. Available: 'triplet', 'prototypical', 'angproto', 'dproto'.
- `train.margin`. Loss margin parameter. 


## Other scripts
- `train_class_loss.py`. To train model for end-to-end classification. 
- `test_supervised_openset.py`. Test few-shot open set the model trained for end-to-end classification. 



## Acknownoledge

We acknowledge the following code repositories:
- https://github.com/ArchitParnami/Few-Shot-KWS
- https://github.com/roman-vygon/triplet_loss_kws
- https://github.com/clovaai/voxceleb_trainer
- https://github.com/BoLiu-SVCL/meta-open/
- https://github.com/tyler-hayes/Embedded-CL
- https://github.com/MrtnMndt/OpenVAE_ContinualLearning
- https://github.com/Codelegant92/STC-ProtoNet

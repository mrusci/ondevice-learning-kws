# On-Device Learning Keyword Spotting (KWS)

This repository includes the experiment code to design and test lightweight keyword spotting models that can learn new keywords over time after deployment on resource-constrained embedded systems, e.g., low-power microcontrollers.

The on-device learning approaches are described in the following papers:

## Few-Shot Open-Set Learning KWS

```
@inproceedings{rusci_interspeech23,
  author={Rusci, Manuele and Tuytelaars, Tinne},
  title={Few-Shot Open-Set Learning for On-Device Customization of KeyWord Spotting Systems},
  year=2023,
  booktitle={Proc. Interspeech},
}
```
```
@article{rusci2023device,
  title={On-device customization of tiny deep learning models for keyword spotting with few examples},
  author={Rusci, Manuele and Tuytelaars, Tinne},
  journal={IEEE Micro},
  year={2023},
  publisher={IEEE}
}
```
These two works focus the problem of on-device customization of KWS model to learn new keywords after deployment, i.e., keywords not known at training time.
The first [paper](https://www.isca-archive.org/interspeech_2023/rusci23_interspeech.pdf) describes a framework to evaluate KWS models that can learn new keywords by recording few utterance examples.
The second [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10241972) illustrates a microcontroller-based system solution for KWS few-shot learning. 

The repository includes the training and test scripts for a KWS audio encoder that can be initialized on-device to recongnize new keywords. 
The official branch of the INTERSPEECH paper is tagged `interspeech23`.
More details on how to setup the code can be found [here](FewShotKWS.md)

## Self-Learning for Personalized KWS
```
@article{rusci2024self,
  title={Self-Learning for Personalized Keyword Spotting on Ultra-Low-Power Audio Sensors},
  author={Rusci, Manuele and Paci, Francesco and Fariselli, Marco and Flamand, Eric and Tuytelaars, Tinne},
  journal={arXiv preprint arXiv:2408.12481},
  year={2024}
}
```
This [work](https://arxiv.org/pdf/2408.12481) describes a method to incrementally fine-tune a KWS model after few-shot initialization. The principle is llustrated in the figure below. After a calibration with respect to the few-shot data, a labeling task assigns pseudo-labels to new unsupervised data based on the similarity with respect to the prototype. The collected pseudo-labeled data are used for the fine-tuning of the model. 

![image](images/selflearningscheme.png)

### Reproducing paper results with Public Data 


The `self_learning_personalized_kws.py` in the `KWSFSL/` folder contains the code of the proposed solution. 
As an example, you can run: 
```
python KWSFSL/self_learning_personalized_kws.py --model_path <pretrained_model_path> --dataset <dataset_name> --pos_selflearn_thr 0.3 --neg_selflearn_thr 0.9 --adapt_set_ratio 0.7 --step_size_ratio 0.125 --train.epochs 20 --train.triplet_type anchor_triplet --data_dir_pos <dataset_pos_path> --data_dir_neg <dataset_neg_path> --log.dirname <dir_name> --log.results_json <json_file_name>
```

* `<dataset_name>`: two options available: `heysnapdragon` or `heysnips`. The datasets can be found at the following links: [HeySnips](https://github.com/sonos/keyword-spotting-research-datasets/tree/master) and [HeySnapdragon](https://developer.qualcomm.com/project/keyword-speech-dataset). For HeySnips, please refer to the latest version: Keyword Spotting Dataset v2 -- "Federated Learning for Keyword Spotting", Leroy et al. (2019).
* `pos_selflearn_thr` and `neg_selflearn_thr` are respectively the positive and negative thresholds. 
* `<dataset_pos_path>`: path to the HeySnips or HeySnapdragon data. 
* `<dataset_neg_path>`: path to the negative data. In our experiment, we always use the HeySnips negative data. If not defined, the  <dataset_neg_path> = <dataset_pos_path>.
* `<pretrained_model_path>`: path to the .pt model to be incrementally fine-tuned. The repository include several pretrained models used for the experiments in the `pretrained_models/` folder: DSCNNS, DSCNNM, DSCNNL, RESNET15. As an example, use pretrained_models/RESNET15.pt for the path. 
* `<dir_name>` and `<json_file_name>` specify where to store the output data, e.g.: --log.dirname logs/public --log.results_json heysnips.json

To parse the output file you can use the script (adjust the dataset name and match the log output name with respect to what you used):
```
python scripts/SelfLearningLogAnalysis.py --dataset heysnips --log_name logs/public/heysnips.json
```
For the setting used in the paper (Tab. III), you can refer to the script `scripts/run_self_learning_public.sh`.

The code has been tested with the following package version:
- torch                1.12.1 
- torchaudio           0.12.1 
- librosa              0.9.2 
- numpy                1.21.6 
- scikit-learn         1.0.2 
- scipy                1.7.3 


### Reproducing paper results with Collected Data
We run experiments with data recorded using the sensor setup in the picture.

![image](images/recording_setup.png)

First, we recorded a dataset of speech samples with our microphone sensor while replaying a subset of data from the HeySnips dataset using the speaker.
The collected data (total of 400 samples) are split between a testset and a trainset, both including "Hey Snips" utterances and non-"Hey Snips" uterrances. 
In particular, the data of the testset is composed by the recordings from 20 random speakers from the original set. 
After recording, the data were fed to our DNN models deployed on devices. 
Initially, a per-speaker prototype vector is computed by feeding three audio recordings of the target keywords. 
Next, the audio tracks of the training set are processed with a sliding window approach to compute the distance with respect to the prototype and assign pseudo-labels for the self-learning task.

These data, which can be found at this [link]() (**dataset is under review for the final publication**), is composed by two main partition:
- *recorded_speech_data*: includes the audio recordings. Note that this dataset is under restricted access to not violate the term of access of the original data.
- *processed_outputs*: includes the output of the processing, i.e. the measured distances.

These data must be downloaded and placed under the same folder, e.g.: \
|-> <ondevice_dataset_path>/ \
|-----> dst/ (recorded_speech_data) \
|-----> logs/ (processed_outputs) \
Make sure to use these folder names to comply with the script `self_learning_personalized_kws_realdata.py` used for the experiments. 
For the setting used in the paper (Fig. 7), you can refer to the script `scripts/run_self_learning_gap.sh`.

As an example, you can run: 
```
python KWSFSL/self_learning_personalized_kws_realdata.py --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --train.triplet_type anchor_triplet --num_pos_batch 10 --num_neg_batch 60 --train.epochs 8 --model_path pretrained_models/RESNET15.pt --ds_config_file res15_ne16 --data_dir <ondevice_dataset_path> --num_experiment 10 
```
where `--ds_config_file` specifies the file of the ondevice measurements. In this examples, _res15_ indicates the model deployed on-device and _ne16_ indicates a processing configuration using the NE16 accelerator. Make sure to indicate the same model in the model path `--model_path`.

The folder of recorded data (*recorded_speech_data*) includes:
- All the recorded audio in numpy formats (.npy). We use the filename of the original file. 
- Multiple speaker folders (spk_xx_initwav, where xx=0,..19) containing the 3 wav files used to compute the prototype per-speaker vector. Similarly, the neg_initwav folder includes the initialization data for the negative prototype. 
- `list.json` file details to which dataset every file belongs to and its label. More in details, the file includes a dictionary with the following info.
```
{ 
  "train":{
    "pos":[<list of positive files for training>],
    "neg":[<list of negative files for training>]
  },
  "test":{
    "spk_00":{
      "init": [<list of initialization files>],
      "test": [<list of positive test files>],
    },
    ....,
    "spk_19":{
      "init": [<list of initialization files>],
      "test": [<list of positive test files>],
    },
    "neg":{
      "init": [<list of initialization files>],
      "test": [<list of negative test files>],
    },
  }
}
```

The measurement files (*processed_outputs*) are multiple files with format <model>_<type_of_processing>, where:
- <model> can be: res15, dscnnl, dscnnm, dscnns
- <type_of_processing> can be: 'ne16' or 'fp16', if the results are obtained by using respectively the NE16 accelerator or the multi-core CPUs (half-precision floating point data precision)



## License
The code is distributed under MIT license. 
Part of the code is however inspired or taken by other repositories as carefully detailed in notice.txt.  


## Acknownoledge
This work is supported by the Horizon Europe program under the Marie-Curie Post-Doctoral Fellowship Program: project SEA2Learn (grant agreement 101067475).
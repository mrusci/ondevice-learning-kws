# resnet
python KWSFSL/self_learning_personalized_kws_realdata.py --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --train.triplet_type anchor_triplet --num_pos_batch 10 --num_neg_batch 60 --train.epochs 8 --model_path pretrained_models/RESNET15.pt --ds_config_file res15_ne16 --data_dir <ondevice_dataset_path> --num_experiment 10 --use_oracle 
python KWSFSL/self_learning_personalized_kws_realdata.py --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --train.triplet_type anchor_triplet --num_pos_batch 10 --num_neg_batch 60 --train.epochs 8 --model_path pretrained_models/RESNET15.pt --ds_config_file res15_ne16 --data_dir <ondevice_dataset_path> --num_experiment 10 

# dscnnl
python KWSFSL/self_learning_personalized_kws_realdata.py --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --train.triplet_type anchor_triplet --num_pos_batch 10 --num_neg_batch 60 --train.epochs 8 --model_path pretrained_models/DSCNNL.pt --ds_config_file dscnnl_ne16 --data_dir <ondevice_dataset_path> --num_experiment 10 --use_oracle 
python KWSFSL/self_learning_personalized_kws_realdata.py --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --train.triplet_type anchor_triplet --num_pos_batch 10 --num_neg_batch 60 --train.epochs 8 --model_path pretrained_models/DSCNNL.pt --ds_config_file dscnnl_ne16 --data_dir <ondevice_dataset_path> --num_experiment 10 

# dscnnm
python KWSFSL/self_learning_personalized_kws_realdata.py --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --train.triplet_type anchor_triplet --num_pos_batch 10 --num_neg_batch 60 --train.epochs 8 --model_path pretrained_models/DSCNNM.pt --ds_config_file dscnnm_ne16 --data_dir <ondevice_dataset_path> --num_experiment 10 --use_oracle 
python KWSFSL/self_learning_personalized_kws_realdata.py --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --train.triplet_type anchor_triplet --num_pos_batch 10 --num_neg_batch 60 --train.epochs 8 --model_path pretrained_models/DSCNNM.pt --ds_config_file dscnnm_ne16 --data_dir <ondevice_dataset_path> --num_experiment 10 

# dscnns
python KWSFSL/self_learning_personalized_kws_realdata.py --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --train.triplet_type anchor_triplet --num_pos_batch 10 --num_neg_batch 60 --train.epochs 8 --model_path pretrained_models/DSCNNS.pt --ds_config_file dscnns_ne16 --data_dir <ondevice_dataset_path> --num_experiment 10 --use_oracle 
python KWSFSL/self_learning_personalized_kws_realdata.py --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --train.triplet_type anchor_triplet --num_pos_batch 10 --num_neg_batch 60 --train.epochs 8 --model_path pretrained_models/DSCNNS.pt --ds_config_file dscnns_ne16 --data_dir <ondevice_dataset_path> --num_experiment 10 
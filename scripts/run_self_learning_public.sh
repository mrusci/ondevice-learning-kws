# commands to reproduce main results (oracle and self-lerning)
python KWSFSL/self_learning_personalized_kws.py --model_path <add_pretrained_mode_path> --dataset <add_dataset> --pos_selflearn_thr 0.3 --neg_selflearn_thr 0.9 --adapt_set_ratio 0.7 --step_size_ratio 0.125 --train.epochs 20 --train.triplet_type anchor_triplet --data_dir_pos <add_path_pos_data> --data_dir_neg <add_path_neg_data>
python KWSFSL/self_learning_personalized_kws.py --model_path <add_pretrained_mode_path> --dataset <add_dataset> --pos_selflearn_thr 0.4 --neg_selflearn_thr 0.9 --adapt_set_ratio 0.7 --step_size_ratio 0.125 --train.epochs 20 --train.triplet_type anchor_triplet --data_dir_pos <add_path_pos_data> --data_dir_neg <add_path_neg_data>
python KWSFSL/self_learning_personalized_kws.py --model_path <add_pretrained_mode_path> --dataset <add_dataset> --step_size_ratio 0.125 --use_oracle --adapt_set_ratio 0.7 --train.epochs 20 --train.triplet_type anchor_triplet --data_dir_pos <add_path_pos_data> --data_dir_neg <add_path_neg_data>
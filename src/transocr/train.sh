# Chinese
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp_name='train-transocr-zh' \
                                            --train_dataset='your/path/to/dataset_zh/{full, upper, lower}/labels_train.txt' \
                                            --test_dataset='your/path/to/dataset_zh/{full, upper, lower}/labels_val.txt' \
                                            --alpha_path ./data/alphabet_zh.txt \
                                            --radical \
                                            --patience 10 

# English
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp_name='train-transocr-en' \
                                            --train_dataset='your/path/to/dataset_en/{full, upper, lower}/labels_train.txt' \
                                            --test_dataset='your/path/to/dataset_en/{full, upper, lower}/labels_val.txt' \
                                            --alpha_path ./data/alphabet_en.txt \
                                            --patience 10 
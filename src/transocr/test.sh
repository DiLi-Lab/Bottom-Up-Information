# Chinese
CUDA_VISIBLE_DEVICES=0,1 python main.py --test \
                                        --exp_name='test-transocr-zh' \
                                        --resume ./history/train-transocr-full-zh-radical-32914/best_model.pth \
                                        --test_dataset 'your/paht/to/dataset_zh/{full, upper, lower}/labels.txt' \
                                        --radical \
                                        --alpha_path ./data/alphabet_zh.txt \

# English
CUDA_VISIBLE_DEVICES=0,1 python main.py --test \
                                        --exp_name='test-transocr-en' \
                                        --resume ./history/train-transocr-upperhalf-en-word-32277/best_model.pth \
                                        --test_dataset 'your/paht/to/dataset_en/{full, upper, lower}/labels.txt' \
                                        --alpha_path ./data/alphabet_en.txt \
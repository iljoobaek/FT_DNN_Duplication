NUM_DUP=32
MODEL=models/attention3/ssd300_COCO_9.pth

#python eval_ssd.py --error_rate 0 --num_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model models/mobilenet-v1-ssd-mp-0_675.pth --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.1 --num_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model models/attention3/ssd300_COCO_9.pth --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.3 --num_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model models/mobilenet-v1-ssd-mp-0_675.pth --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.5 --num_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model models/mobilenet-v1-ssd-mp-0_675.pth --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.7 --num_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model models/mobilenet-v1-ssd-mp-0_675.pth --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.9 --num_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model models/mobilenet-v1-ssd-mp-0_675.pth --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 1 --num_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model models/mobilenet-v1-ssd-mp-0_675.pth --label_file models/voc-model-labels.txt

#python eval_ssd.py --error_rate 0.1 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.3 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.5 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.7 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.9 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 1 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt

#python eval_ssd.py --error_rate 0.1 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.3 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.5 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.7 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 0.9 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#python eval_ssd.py --error_rate 1 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
#

echo #duplication=${NUM_DUP} >> result.txt
python eval_ssd.py --error_rate 0.1 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
python eval_ssd.py --error_rate 0.3 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
python eval_ssd.py --error_rate 0.5 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
python eval_ssd.py --error_rate 0.7 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
python eval_ssd.py --error_rate 0.9 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt
python eval_ssd.py --error_rate 1 --num_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${MODEL} --label_file models/voc-model-labels.txt




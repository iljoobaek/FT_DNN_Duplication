export CUDA_VISIBLE_DEVICES=1
NUM_DUP=0.5
IDX=1
MODEL=models/attention3/ssd300_COCO_9.pth
ATMODEL=models/attention3/${IDX}_ssd300_COCO_9.pth
#python eval_ssd.py --error_rate 0.1 --percent_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX} --weight_error 0.01

#python eval_ssd.py --error_rate 0.1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX} --weight_error 0.01

#python eval_ssd.py --error_rate 0.1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX} --weight_error 0.01

#python eval_ssd.py --error_rate 0.1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type random --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX} --weight_error 0.01

python eval_ssd.py --error_rate 0.1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type entropy --dataset_type voc --dataset /home/droid/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX} --weight_error 0.01

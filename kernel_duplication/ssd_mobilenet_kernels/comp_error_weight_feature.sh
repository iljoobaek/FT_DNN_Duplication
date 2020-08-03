export CUDA_VISIBLE_DEVICES=1
IDX=1
NUM_DUP=0.5
ATMODEL=models/attention3/${IDX}_ssd300_COCO_9.pth
ERR=0

for WERR in $(seq 0.9 0.1 1)
do
    python eval_ssd.py --error_rate ${ERR} --percent_duplication ${NUM_DUP} --run_original False --duplicated False --ft_type None --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX} --weight_error ${WERR}
done
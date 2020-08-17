NUM_DUP=0.5
IDX=1
MODEL=models/attention3/ssd300_COCO_9.pth
DATAPATH=/home/droid/data/VOCdevkit/VOC2007
SAVEFILE=result_full_dup.txt

ATMODEL=models/attention3/${IDX}_ssd300_COCO_9.pth
for WERR in $(seq 0.01 0.01 0.01)
do
echo "error=${WERR}" >> result.txt
for ERR in $(seq 0.01 0.01 0.1)
do
python eval_ssd.py --error_rate ${ERR} --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset ${DATAPATH} --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX} --weight_error ${WERR} --result_save_file ${SAVEFILE}

done
done
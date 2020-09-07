export CUDA_VISIBLE_DEVICES=1
NUM_DUP=0.5
IDX=1
MODEL=models/attention3/ssd300_COCO_9.pth
DATAPATH=/home/rtml/data/VOCdevkit/VOC2007
SAVEFILE=result_d2nn_dup.txt

ATMODEL=models/attention3/${IDX}_ssd300_COCO_9.pth

echo "Entropy" >> ${SAVEFILE}
#for NUM_DUP in $(seq 0.01 0.02 0.09)
#do
for WERR in $(seq 0.01 0.01 0.01)
# WERR=0.02
# for NUM_DUP in $(seq 0.1 0.2 0.1)
do
# echo "error=${WERR}" >> ${SAVEFILE}
echo "Error=${WERR}, Dup percentage=${NUM_DUP}" >> ${SAVEFILE}
for ERR in $(seq 0.01 0.01 0.1)
do
python eval_ssd.py --error_rate ${ERR} --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type entropy_p --dataset_type voc --dataset ${DATAPATH} --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX} --weight_error ${WERR} --result_save_file ${SAVEFILE}

done
done
#done

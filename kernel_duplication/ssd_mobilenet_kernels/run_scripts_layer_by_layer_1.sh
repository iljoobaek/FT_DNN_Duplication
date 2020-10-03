CUDA_VISIBLE_DEVICES=1

NUM_DUP=0.5
IDX=1
MODEL=models/attention3/ssd300_COCO_9.pth
# DATA=/home/rtml/data/VOCdevkit/VOC2007
DATA=/home/droid/Documents/data/VOCdevkit/VOC2007

WERR=0
# for WERR in $(seq 0.02 0.01 0.1)
# do
# echo "error=${WERR}" >> result.txt
for SEED in $(seq 3 1 3)
do
echo "seed=${SEED}" >> result.txt
for IDX in $(seq 1 1 12)
do
echo "idx=${IDX}" >> result.txt
ATMODEL=models/attention3/${IDX}_ssd300_COCO_9.pth
for ERR in $(seq 0.1 0.4 0.5)
do
echo "err=${ERR}" >> result.txt
python eval_ssd.py --error_rate ${ERR} \
                   --percent_duplication 0 \
                   --run_original False \
                   --duplicated False \
                   --ft_type none \
                   --dataset_type voc \
                   --dataset ${DATA} \
                   --net mb1-ssd \
                   --trained_model ${ATMODEL} \
                   --label_file models/voc-model-labels.txt \
                   --weight_index ${IDX} \
                   --weight_error ${WERR} \
                   --seed ${SEED}

python eval_ssd.py --error_rate ${ERR} \
                   --percent_duplication ${NUM_DUP} \
                   --run_original False \
                   --duplicated True \
                   --ft_type attention \
                   --dataset_type voc \
                   --dataset ${DATA} \
                   --net mb1-ssd \
                   --trained_model ${ATMODEL} \
                   --label_file models/voc-model-labels.txt \
                   --weight_index ${IDX} \
                   --weight_error ${WERR} \
                   --seed ${SEED}

python eval_ssd.py --error_rate ${ERR} \
                   --percent_duplication ${NUM_DUP} \
                   --run_original False \
                   --duplicated True \
                   --ft_type importance \
                   --dataset_type voc \
                   --dataset ${DATA} \
                   --net mb1-ssd \
                   --trained_model ${ATMODEL} \
                   --label_file models/voc-model-labels.txt \
                   --weight_index ${IDX} \
                   --weight_error ${WERR} \
                   --seed ${SEED}

python eval_ssd.py --error_rate ${ERR} \
                   --percent_duplication ${NUM_DUP} \
                   --run_original False \
                   --duplicated True \
                   --ft_type d2nn \
                   --dataset_type voc \
                   --dataset ${DATA} \
                   --net mb1-ssd \
                   --trained_model ${ATMODEL} \
                   --label_file models/voc-model-labels.txt \
                   --weight_index ${IDX} \
                   --weight_error ${WERR} \
                   --seed ${SEED}

python eval_ssd.py --error_rate ${ERR} \
                   --percent_duplication ${NUM_DUP} \
                   --run_original False \
                   --duplicated True \
                   --ft_type random \
                   --dataset_type voc \
                   --dataset ${DATA} \
                   --net mb1-ssd \
                   --trained_model ${ATMODEL} \
                   --label_file models/voc-model-labels.txt \
                   --weight_index ${IDX} \
                   --weight_error ${WERR} \
                   --seed ${SEED}
done
done
done
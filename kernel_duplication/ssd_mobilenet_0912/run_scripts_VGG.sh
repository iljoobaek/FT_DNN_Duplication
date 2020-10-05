export CUDA_VISIBLE_DEVICES=1
NUM_DUP=0.5
IDX=1
# MODEL=models/attention3/ssd300_COCO_9.pth
MODEL=models/vgg16-ssd-mp-0_7726.pth
DATAPATH=/home/rtml/data/VOCdevkit/VOC2007
SAVEFILE=result_d2nn_dup_vgg.txt

ATMODEL=models/mobilenet-v1-ssd-mp-0_675.pth

echo "Entropy" >> ${SAVEFILE}
#for NUM_DUP in $(seq 0.01 0.02 0.09)
#do
for SEED in $(seq 1 1 3)
do
for WERR in $(seq 0.1 0.01 0.1)
# WERR=0.02
# for NUM_DUP in $(seq 0.1 0.2 0.1)
do
# echo "error=${WERR}" >> ${SAVEFILE}
# echo "Error=${WERR}, Dup percentage=${NUM_DUP}, Type=No dup, Seed=${SEED}" >> ${SAVEFILE}
# for ERR in $(seq 0.01 0.01 0.1)
# do
# python eval_ssd.py --error_rate ${ERR} \
#                    --percent_duplication ${NUM_DUP} \
#                    --run_original False \
#                    --duplicated False \
#                    --ft_type None \
#                    --dataset_type voc \
#                    --dataset ${DATAPATH} \
#                    --net vgg16-ssd \
#                    --trained_model ${MODEL} \
#                    --label_file models/voc-model-labels.txt \
#                    --weight_index ${IDX} \
#                    --weight_error ${WERR} \
#                    --result_save_file ${SAVEFILE} \
#                    --seed ${SEED}
# done

# echo "Error=${WERR}, Dup percentage=${NUM_DUP}, Type=Random" >> ${SAVEFILE}
# for ERR in $(seq 0.01 0.01 0.1)
# do
# python eval_ssd.py --error_rate ${ERR} \
#                    --percent_duplication ${NUM_DUP} \
#                    --run_original False \
#                    --duplicated True \
#                    --ft_type random \
#                    --dataset_type voc \
#                    --dataset ${DATAPATH} \
#                    --net vgg16-ssd \
#                    --trained_model ${MODEL} \
#                    --label_file models/voc-model-labels.txt \
#                    --weight_index ${IDX} \
#                    --weight_error ${WERR} \
#                    --result_save_file ${SAVEFILE} \
#                    --seed ${SEED}

# done

# echo "Error=${WERR}, Dup percentage=${NUM_DUP}, Type=D2NN" >> ${SAVEFILE}
# for ERR in $(seq 0.01 0.01 0.1)
# do
# python eval_ssd.py --error_rate ${ERR} \
#                    --percent_duplication ${NUM_DUP} \
#                    --run_original False \
#                    --duplicated True \
#                    --ft_type d2nn \
#                    --dataset_type voc \
#                    --dataset ${DATAPATH} \
#                    --net vgg16-ssd \
#                    --trained_model ${MODEL} \
#                    --label_file models/voc-model-labels.txt \
#                    --weight_index ${IDX} \
#                    --weight_error ${WERR} \
#                    --result_save_file ${SAVEFILE} \
#                    --seed ${SEED}

# done

# echo "Error=${WERR}, Dup percentage=${NUM_DUP}, Type=Importance" >> ${SAVEFILE}
# for ERR in $(seq 0.01 0.01 0.1)
# do
# python eval_ssd.py --error_rate ${ERR} \
#                    --percent_duplication ${NUM_DUP} \
#                    --run_original False \
#                    --duplicated True \
#                    --ft_type importance \
#                    --dataset_type voc \
#                    --dataset ${DATAPATH} \
#                    --net vgg16-ssd \
#                    --trained_model ${MODEL} \
#                    --label_file models/voc-model-labels.txt \
#                    --weight_index ${IDX} \
#                    --weight_error ${WERR} \
#                    --result_save_file ${SAVEFILE} \
#                    --seed ${SEED}
# done

echo "Error=${WERR}, Dup percentage=${NUM_DUP}, Type=Entropy" >> ${SAVEFILE}
for ERR in $(seq 0.01 0.01 0.1)
do
python eval_ssd.py --error_rate ${ERR} \
                   --percent_duplication ${NUM_DUP} \
                   --run_original False \
                   --duplicated True \
                   --ft_type entropy_p \
                   --dataset_type voc \
                   --dataset ${DATAPATH} \
                   --net vgg16-ssd \
                   --trained_model ${MODEL} \
                   --label_file models/voc-model-labels.txt \
                   --weight_index ${IDX} \
                   --weight_error ${WERR} \
                   --result_save_file ${SAVEFILE} \
                   --seed ${SEED}
done

done
done
#done

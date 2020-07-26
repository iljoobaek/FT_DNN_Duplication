export CUDA_VISIBLE_DEVICES=1
IDX=12

for IDX in {1..11}
do
    python train_ssd.py --dataset_type voc \
            --datasets /home/rtml/data/VOCdevkit/VOC2007 \
            --validation_dataset /home/rtml/data/VOCdevkit/VOC2007 \
            --net mb1-ssd \
            --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth \
            --checkpoint_folder models \
            --scheduler cosine \
            --lr 0.01 \
            --batch_size 32 \
            --t_max 200 \
            --validation_epochs 50 \
            --num_epochs 10 \
            --run_original False \
            --use_cuda True \
            --weight_index ${IDX}
done


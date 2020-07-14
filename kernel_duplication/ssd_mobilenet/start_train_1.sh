python train_ssd_1.py --dataset_type voc \
        --datasets /home/rtml/data/VOCdevkit/VOC2007 /home/rtml/data/VOCdevkit/VOC2012 \
        --validation_dataset /home/rtml/data/VOCdevkit/VOC2007 \
        --net mb1-ssd \
        --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth \
        --checkpoint_folder models/ssd_mobileV1 \
        --scheduler cosine \
        --lr 0.01 \
        --batch_size 32 \
        --t_max 200 \
        --validation_epochs 50 \
        --num_epochs 40000 \


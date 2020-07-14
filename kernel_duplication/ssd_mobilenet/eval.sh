python eval_ssd.py --dataset_type voc \
        --dataset /home/rtml/data/VOCdevkit/VOC2007 \
        --net mb1-ssd \
        --trained_model models/mobilenet-v1-ssd-mp-0_675.pth \
	--label_file models/ssd_mobileV1/voc-model-labels.txt

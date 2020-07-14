python train_ssd.py --dataset_type voc \
      	--datasets /home/karan/data/VOCdevkit2007/VOC2007 \
	--validation_dataset /home/karan/data/VOCdevkit2007/VOC2007 \
	--net mb2-ssd-lite \
       	--base_net models/mb2-imagenet-71_8.pth \
	--resume models/ssd_mobile2/mb2-ssd-lite-Epoch-600-Loss-1.879618007203807.pth \
	--checkpoint_folder models/ssd_low_loss \
      	--scheduler cosine \
	--lr 0.01 \
	--batch_size 24 \
	--t_max 200 \
	--validation_epochs 100 \
	--num_epochs 40000 \

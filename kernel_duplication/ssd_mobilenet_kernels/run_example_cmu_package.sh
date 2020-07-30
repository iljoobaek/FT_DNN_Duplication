#CUDA_VISIBLE_DEVICES=1 python run_ssd_example_package.py mb2-ssd-lite models/ssd_mobilenet2/mb2-ssd-lite-Epoch-200-Loss-2.273608135653066.pth models/voc-model-labels.txt ../CMU_dataset/set04/ ../CMU_dataset/mobilenet2/set04/

MODEL=models/attention3/1_ssd300_COCO_9.pth

CUDA_VISIBLE_DEVICES=1
python run_ssd_example_package.py mb1-ssd \
                                  ${MODEL} \
                                  models/voc-model-labels.txt \
                                  /home/rtml/data/VOCdevkit/VOC2007 \
                                  detection/

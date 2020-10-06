NUM_DUP=0.5
IDX=11
MODEL=models/attention3/ssd300_COCO_9.pth

for IDX in {2..12}
do
    echo "Layer: ${IDX}" >> result.txt
    ATMODEL=models/attention3/${IDX}_ssd300_COCO_9.pth
    #python eval_ssd.py --error_rate 0 --percent_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.1 --percent_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.3 --percent_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.5 --percent_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.7 --percent_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.9 --percent_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 1 --percent_duplication 0 --run_original False --duplicated False --ft_type none --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}

    #python eval_ssd.py --error_rate 0.1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.3 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.5 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.7 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.9 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type attention --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}

    #python eval_ssd.py --error_rate 0.1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    python eval_ssd.py --error_rate 0.3 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.5 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    python eval_ssd.py --error_rate 0.7 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    python eval_ssd.py --error_rate 0.9 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    python eval_ssd.py --error_rate 1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type importance --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}


    #echo #duplication=${NUM_DUP} >> result.txt
    #python eval_ssd.py --error_rate 0.1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.3 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.5 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.7 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.9 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type d2nn --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}


    #python eval_ssd.py --error_rate 0.1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type random --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.3 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type random --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.5 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type random --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.7 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type random --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 0.9 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type random --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
    #python eval_ssd.py --error_rate 1 --percent_duplication ${NUM_DUP} --run_original False --duplicated True --ft_type random --dataset_type voc --dataset /home/rtml/data/VOCdevkit/VOC2007 --net mb1-ssd --trained_model ${ATMODEL} --label_file models/voc-model-labels.txt --weight_index ${IDX}
done

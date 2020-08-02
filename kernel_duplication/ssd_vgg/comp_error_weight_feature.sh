export CUDA_VISIBLE_DEVICES=0
IDX=1

for ERR in $(seq 0 0.1 1)
do
    python eval_attention_kernel.py --attention_mode False --error_rate ${ERR} --touch_layer_index 1 --weight_index ${IDX} --weight_error 0
done
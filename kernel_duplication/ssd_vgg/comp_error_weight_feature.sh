export CUDA_VISIBLE_DEVICES=1
IDX=5

for ERR in $(seq 0 0.01 1)
do
    python eval_attention_kernel.py --attention_mode False --error_rate 0 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
done
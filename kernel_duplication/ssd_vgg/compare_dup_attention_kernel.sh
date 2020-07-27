export CUDA_VISIBLE_DEVICES=1

# vgg: 0 2 5 7 10 12 14 17 19 21 24 26 28 31
IDX=5

for ERR in 0 0.05 0.1
do
    echo "Err: ${ERR}" >> results
    for IDX in {1..13}
    do
        echo "Layer: ${IDX}" >> results
        python eval_attention_kernel.py --attention_mode False --error_rate 0 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode False --error_rate 0.1 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode False --error_rate 0.3 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode False --error_rate 0.5 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode False --error_rate 0.7 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode False --error_rate 0.9 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode False --error_rate 0.92 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode False --error_rate 0.94 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode False --error_rate 0.96 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode False --error_rate 0.98 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode False --error_rate 1 --touch_layer_index 1 --weight_index ${IDX} --weight_error ${ERR}

        #python eval_attention_kernel.py --attention_mode True --error_rate 0.1 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.3 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.5 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.7 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.9 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.92 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.94 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.96 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.98 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 1 --touch_layer_index 1 --ft_type attention --weight_index ${IDX} --weight_error ${ERR}
        ##
        python eval_attention_kernel.py --attention_mode True --error_rate 0.1 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.3 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.5 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.7 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.9 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.92 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.94 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.96 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.98 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 1 --touch_layer_index 1 --ft_type importance --weight_index ${IDX} --weight_error ${ERR}
        ##
        python eval_attention_kernel.py --attention_mode True --error_rate 0.1 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.3 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.5 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.7 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.9 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.92 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.94 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.96 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.98 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 1 --touch_layer_index 1 --ft_type d2nn --weight_index ${IDX} --weight_error ${ERR}

        python eval_attention_kernel.py --attention_mode True --error_rate 0.1 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.3 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.5 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.7 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 0.9 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.92 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.94 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.96 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
        #python eval_attention_kernel.py --attention_mode True --error_rate 0.98 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
        python eval_attention_kernel.py --attention_mode True --error_rate 1 --touch_layer_index 1 --ft_type random --weight_index ${IDX} --weight_error ${ERR}
    done
done

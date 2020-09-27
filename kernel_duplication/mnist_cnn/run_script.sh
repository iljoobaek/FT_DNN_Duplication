#!/bin/bash
for SEED in $(seq 1 1 1)
do
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 0.1 --seed ${SEED}
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 0.3 --seed ${SEED}
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 0.5 --seed ${SEED}
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 0.7 --seed ${SEED}
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 0.9 --seed ${SEED}
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 0.92 --seed ${SEED}
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 0.94 --seed ${SEED}
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 0.96 --seed ${SEED}
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 0.98 --seed ${SEED}
    python3 mnist_kernel_duplication_subflow.py --evaluate True --error_rate 1 --seed ${SEED}
done

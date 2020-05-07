python3.5 origin_mnist.py

# create the "model" folder to save the model
mkdir model
cd model
mkdir original
mkdir attention

# Train the original model without attention: 
python3.5 mnist_kernel_duplication.py --run_original True

# Train the model with attention : 
python3.5 mnist_kernel_duplication.py --run_original False

# After training, evaluate model's accuracy with vs without attention: 
# change the error rate every time.

python3.5 mnist_kernel_duplication.py --evaluate True --error_rate 0.95


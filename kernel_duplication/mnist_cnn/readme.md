python origin_mnist.py

# create the "model" folder to save the model
mkdir model
cd model
mkdir original
mkdir attention

Train the original model: python mnist_error.py --run_original True
Train the attention model: python mnist_error.py --run_original False

Get evaluation results for both model (with and without attention): python mnist_error.py --evaluate True --error_rate 0.95

# hyper parameters and all other kind of params

# parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 2

# network parameters
n_hidden_1 = 300  # 1st layer number of neurons
n_hidden_2 = 300  # 2nd layer number of neurons
num_input = 784   # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

# Training Parameters
checkpoint_every = 100
checkpoint_dir = './runs/'

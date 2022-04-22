# reminder... in vscode, select some lines and then hit shift-enter to execute

# let's download/install the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# let's grab an image and plot it
sample_image = mnist.train.images[0].reshape([28,28])
import matplotlib.pyplot as plt
plt.gray()
plt.imshow(sample_image)
plt.show()



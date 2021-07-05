import time

import tensorflow as tf

"""GPU acceleration
Many TensorFlow operations are accelerated using the GPU for computation. Without any 
annotations, TensorFlow automatically decides whether to use the GPU or CPU for an 
operation—copying the tensor between CPU and GPU memory, if necessary. Tensors produced 
by an operation are typically backed by the memory of the device on which the operation 
executed, for example:
"""

"""
To enable tensorflow use GPU:
1. You need a gpu driver(normally you already have)
2. You need to install cuda tool kit.

The full installation guide of cuda tool kit can be found here:

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Four ubuntu: 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

"""


def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time() - start

    print("10 loops: {:0.2f}ms".format(1000 * result))


def exp1():
    x = tf.random.uniform([3, 3])
    print("Is there a GPU available: "),
    # list all available GPU
    print(tf.config.list_physical_devices("GPU"))

    # check if a tensor is running on GPU #0
    print("Is the Tensor on GPU #0:  ")
    print(x.device.endswith('GPU:0'))


def exp2():
    # Force execution on CPU
    print("On CPU:")
    with tf.device("CPU:0"):
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith("CPU:0")
        time_matmul(x)

    # Force execution on GPU #0 if available
    if tf.config.list_physical_devices("GPU"):
        print("On GPU:")
        with tf.device("GPU:0"):  # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
            x = tf.random.uniform([1000, 1000])
            assert x.device.endswith("GPU:0")
            time_matmul(x)


def main():
    # In exp1, we show the list of available gpu, and if a tensor is running a gpu
    exp1()

    # In exp2, we show how to force tensorflow execution on a device (e.g. cpu, gpu)
    exp2()


if __name__ == "__main__":
    main()

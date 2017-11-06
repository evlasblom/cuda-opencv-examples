# CUDA & Open CV Examples

Many examples exist for using ready-to-go CUDA implementations of algorithms in Open CV. But what if you want to start writing your own CUDA kernels in combination with already existing functionality in Open CV? This repository demonstrates several examples to do just that. Before doing so,  it is recommended to at least go through the first half of the [CUDA basics](http://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf). More information is provided in the comments of the examples.

## Content
The examples start out simple with an empty kernel and gradually become more difficult until the point where we are able to manipulate vectors of Open CV Mat objects on the GPU. It is recommended to go through them in the order as presented below.

__Basic Kernels:__
1. hello.cu: our first kernel.
2. add.cu: basic kernel for parallel additions.
3. ptp.cu: smart usage of pointers on the device.

__Image Processing Kernels:__
1. bgrtogray.cu: our first image processing kernel.
2. invert_1.cu: image invert using low-level operations.
3. invert_2.cu: image invert using high-level Open CV objects.

__Advanced Usage:__
1. diff_1.cu: image differencing using high-level Open CV objects.
2. diff_2.cu: image differencing with smart usage of pointers.
1. conversions.cpp: conversions between high-level Open CV objects.
2. diff_proper.cu: including the kernel template and operation overloading.
3. split.cu: image splitting by combining predefined and custom kernels.

## Requirements
These examples require Open CV and CUDA to be installed on your system. The examples were tested with Open CV 3.2, CUDA 8.0 on Ubuntu 14.04 LTS.

## Building
To build the examples is done as follows.

```bash
cd build
cmake ..
make
```
/**
 * @brief      Image differencing.
 *
 *             In this example we again upload multiple input matrices to the
 *             GPU. However, here we do not upload the matrices separately. This
 *             would be a lot of code duplication for a larger amount of images.
 *             Plus, having the ability to work on a variable amount of images
 *             increases the flexibility of the kernel. To achieve this, we
 *             apply the method of smart pointers to Open CV objects. This is
 *             done by creating a GpuMat container that in itself contains data
 *             that points to the first element of other matrices, similar to
 *             the ptp.cu example shown earlier.
 *
 *             From:
 *             https://github.com/opencv/opencv/blob/master/modules/cudafeatures2d/src/brute_force_matcher.cpp#L130
 *
 *             Built-in data types from:
 *             http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
 */
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp> 
#include <opencv2/core/cuda/common.hpp>
#include "device_launch_parameters.h" // for linting


/// global variables
std::vector<cv::cuda::GpuMat> ginputs;
cv::cuda::GpuMat goutput;

/// number of input matrices
#define N 2

/**
 * @brief      The difference kernel.
 *
 * @param[in]  inputs  The inputs
 * @param[in]  n       The number of inputs
 * @param[in]  output  The output
 *
 * @tparam     T       The type
 */
template <typename T>
__global__ void diff_kernel(const cv::cuda::PtrStepSz<T>* inputs, int n, cv::cuda::PtrStepSz<T> output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const cv::cuda::PtrStepSz<T> input1 = inputs[0];
  const cv::cuda::PtrStepSz<T> input2 = inputs[1];
  T v1 = input1(y, x);
  T v2 = input2(y, x);
  T result;
  result.x = v2.x - v1.x;
  result.y = v2.y - v1.y;
  result.z = v2.z - v1.z;
  output(y, x) = result; 
}

/**
 * @brief      Uploads the inputs.
 *
 *             Implementation based on:
 *             https://github.com/opencv/opencv/blob/master/modules/cudafeatures2d/src/brute_force_matcher.cpp#L130
 *
 * @param[in]  vCollection  The vectorized collection
 * @param      gCollection  The gpu collection
 */
static void upload_inputs(const std::vector<cv::cuda::GpuMat>& vCollection, cv::cuda::GpuMat& gCollection) {
  if (vCollection.empty()) return;

  cv::Mat gCollectionCPU(1, static_cast<int>(vCollection.size()), CV_8UC(sizeof(cv::cuda::PtrStepSzb)));

  cv::cuda::PtrStepSzb* gCollectionCPU_ptr = gCollectionCPU.ptr<cv::cuda::PtrStepSzb>();

  for (size_t i = 0, size = vCollection.size(); i < size; ++i, ++gCollectionCPU_ptr)
      *gCollectionCPU_ptr = vCollection[i];

  gCollection.upload(gCollectionCPU);
}

/**
 * @brief      Calls the difference kernel.
 *
 * @param[in]  inputs  The inputs
 * @param      output  The output
 *
 * @tparam     T       The type
 */
template <typename T>
void call_diff_kernel(const cv::cuda::PtrStepSzb& inputs, cv::cuda::GpuMat& output) {
  	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid(cv::cuda::device::divUp(output.cols, block.x), cv::cuda::device::divUp(output.rows, block.y));

	// Launch kernel
  	diff_kernel<T><<<grid, block>>>((cv::cuda::PtrStepSz<T>*)inputs.ptr(), inputs.cols, static_cast<cv::cuda::PtrStepSz<T>>(output));

  	// Get last error
  	cudaSafeCall(cudaGetLastError());
}

/**
 * @brief      Initializes the difference kernel
 *
 *             Performs the required memory allocation on the CPU and GPU.
 *
 * @param[in]  input1  The input 1
 * @param[in]  input2  The input 2
 * @param      output  The output
 */
void diff_kernel_init(const cv::Mat& input1, const cv::Mat& input2, cv::Mat& output) {
	ginputs.reserve(N);
	ginputs.push_back(cv::cuda::GpuMat(input1.rows, input1.cols, input1.type()));
	ginputs.push_back(cv::cuda::GpuMat(input2.rows, input2.cols, input2.type()));
  	goutput.create(output.rows, output.cols, output.type());
}

/**
 * @brief      Wrapper for the difference kernel.
 *
 * @param[in]  input1  The input 1
 * @param[in]  input2  The input 2
 * @param[in]  output  The output
 */
void diff_kernel_exec(const cv::Mat& input1, const cv::Mat& input2, cv::Mat& output) {
	// upload
	cv::cuda::GpuMat gCollection;
	upload_inputs(ginputs, gCollection);
  	ginputs[0].upload(input1);
  	ginputs[1].upload(input2);

  	// execute
  	call_diff_kernel<char3>(gCollection, goutput);    
  
  	// download
  	goutput.download(output);
}

/**
 * @brief      Terminates the difference kernel
 *
 *             Performs the required memory cleanup on the CPU and GPU.
 *
 * @param[in]  input1  The input 1
 * @param[in]  input2  The input 2
 * @param      output  The output
 */
void diff_kernel_exit(const cv::Mat& input1, const cv::Mat& input2, cv::Mat& output) {
	ginputs[0].release();
	ginputs[1].release();
	ginputs.clear();
	goutput.release();
}


// ----------------------------------------------------------------------


int main() {
	// Read input image from the disk
	std::string imagePath = "../data/image.jpg";
	cv::Mat input1 = cv::imread(imagePath,CV_LOAD_IMAGE_COLOR);

	if(input1.empty())	{
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}

	cv::Mat input2 = cv::Mat::zeros(input1.size(), input1.type());
    input1(cv::Rect(0,10, input1.cols,input1.rows-10)).copyTo(input2(cv::Rect(0,0,input1.cols,input1.rows-10)));

	// Create output image
	cv::Mat output = cv::Mat::zeros(input1.size(), input1.type());

	// Call the wrapper function
	diff_kernel_init(input1, input2, output);
	diff_kernel_exec(input1, input2, output);
	diff_kernel_exit(input1, input2, output);

	// Show the input and output
	cv::imshow("Input1",input1);
	cv::imshow("Input2",input2);
	cv::imshow("Output",output);
	
	// Wait for key press
	cv::waitKey();

	return 0;
}
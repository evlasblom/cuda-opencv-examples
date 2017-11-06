/**
 * @brief      Image splitting.
 *
 *             In this final example make again use of pointers to allocate a
 *             variable amount of matrices on the device. Though, this time, it
 *             is used to produce a variable amount of outputs. In addition, we
 *             show how to manipulate the same parts of the memory (as uploaded
 *             with the pointer containers), but now using predefined CUDA
 *             operations in Open CV.
 *
 *             This finally concludes all examples. We are now able to upload
 *             images with various underlying data types, with a variable amount
 *             of inputs and outputs, making use of high-level data containers
 *             from Open CV for ease-of-use and combining custom kernels with
 *             predefined kernels on the same data.
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


/**
 * @brief      Sets the value of a uchar type.
 *
 * @param[in]  val   The value
 * @param      out   The output
 */
__device__ __forceinline__ void set_value(const int& val, uchar& out) {
    out = val;
}

/**
 * @brief      Sets the value of a uchar3 type.
 *
 * @param[in]  val   The value
 * @param      out   The output
 */
__device__ __forceinline__ void set_value(const int& val, uchar3& out) {
    out = make_uchar3(val, val, val);
}

/**
 * @brief      Sets the value of a char type.
 *
 * @param[in]  val   The value
 * @param      out   The output
 */
__device__ __forceinline__ void set_value(const int& val, char& out) {
    out = val;
}

/**
 * @brief      Sets the value of a char3 type.
 *
 * @param[in]  val   The value
 * @param      out   The output
 */
__device__ __forceinline__ void set_value(const int& val, char3& out) {
    out = make_char3(val, val, val);
}

/**
 * @brief      Subtraction for uchar types.
 *
 * @param[in]  in1   Input 1
 * @param[in]  in2   Input 2
 *
 * @return     Output
 */
__device__ __forceinline__ uchar subtract_value(uchar in1, uchar in2) {
	return in2-in1;
}

/**
 * @brief      Subtraction for uchar3 types.
 *
 * @param[in]  in1   Input 1
 * @param[in]  in2   Input 2
 *
 * @return     Output
 */
__device__ __forceinline__ uchar3 subtract_value(uchar3 in1, uchar3 in2) {
  return make_uchar3(in2.x-in1.x, in2.y-in1.y,in2.z-in1.z);
}

/**
 * @brief      Subtraction for char types.
 *
 * @param[in]  in1   Input 1
 * @param[in]  in2   Input 2
 *
 * @return     Output
 */
__device__ __forceinline__ char subtract_value(char in1, char in2) {
	return in2-in1;
}

/**
 * @brief      Subtraction for char3 types.
 *
 * @param[in]  in1   Input 1
 * @param[in]  in2   Input 2
 *
 * @return     Output
 */
__device__ __forceinline__ char3 subtract_value(char3 in1, char3 in2) {
  return make_char3(in2.x-in1.x, in2.y-in1.y,in2.z-in1.z);
}

/**
 * @brief      The split kernel.
 *
 * @param[in]  inputs  The inputs
 * @param[in]  n       The number of inputs
 * @param[in]  output  The output
 *
 * @tparam     T       The type
 */
template <typename T>
__global__ void split_kernel(cv::cuda::PtrStepSz<T>* const goutputs, int n, const cv::cuda::PtrStepSz<T> input) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  for (int i = 0; i < n; ++i) {
  	if ((y <= input.rows*(i+1)/n) && (y > input.rows*(i)/n)) {
  		goutputs[i](y, x) = input(y, x);
  	}
  }
}

/**
 * @brief      Uploads the ouputs.
 *
 *             Implementation based on:
 *             https://github.com/opencv/opencv/blob/master/modules/cudafeatures2d/src/brute_force_matcher.cpp#L130
 *
 * @param[in]  outputs             The outputs
 * @param      goutputs            The goutputs
 * @param      goutput_collection  The gpu collection
 * @param[in]  vCollection  The vectorized collection
 */
void upload_outputs(std::vector<cv::Mat>& outputs, std::vector<cv::cuda::GpuMat>& goutputs, cv::cuda::GpuMat& goutput_collection) {
  if (goutputs.empty()) return;

  cv::Mat goutput_collectionCPU(1, static_cast<int>(goutputs.size()), CV_8UC(sizeof(cv::cuda::PtrStepSzb)));
  cv::cuda::PtrStepSzb* goutput_collectionCPU_ptr = goutput_collectionCPU.ptr<cv::cuda::PtrStepSzb>();

  for (int i = 0, size = goutputs.size(); i < size; ++i, ++goutput_collectionCPU_ptr) {
    *goutput_collectionCPU_ptr = goutputs[i];
    goutputs[i].upload(outputs[i]);
  }

  goutput_collection.upload(goutput_collectionCPU);
}

/**
 * @brief      Calls the invert kernel.
 *
 * @param[in]  inputs  The inputs
 * @param      output  The output
 *
 * @tparam     T       The type
 */
template <typename T>
void call_split_kernel(cv::cuda::PtrStepSzb outputs, const cv::cuda::GpuMat& input) {
  	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid(cv::cuda::device::divUp(input.cols, block.x), cv::cuda::device::divUp(input.rows, block.y));

	// Launch kernel
  	split_kernel<T><<<grid, block>>>((cv::cuda::PtrStepSz<T>*)outputs.ptr(), outputs.cols, static_cast<cv::cuda::PtrStepSz<T>>(input));

  	// Get last error
  	cudaSafeCall(cudaGetLastError());
}

/**
 * @brief      Initializes the invert kernel
 *
 *             Performs the required memory allocation on the CPU and GPU.
 *
 * @param      input     The inputs
 * @param      ginput    The ginput
 * @param      outputs   The output
 * @param      goutputs  The goutputs
 */
void split_kernel_init(cv::Mat& input, cv::cuda::GpuMat& ginput, std::vector<cv::Mat>& outputs, std::vector<cv::cuda::GpuMat>& goutputs) {
	goutputs.reserve(outputs.size());
	for (int i = 0; i < outputs.size(); ++i)	{
		goutputs.push_back(cv::cuda::GpuMat(outputs[i].rows, outputs[i].cols, outputs[i].type()));
	}

  	ginput.create(input.rows, input.cols, input.type());
}

/**
 * @brief      Wrapper for the invert kernel.
 *
 * @param      input     The inputs
 * @param      ginput    The ginput
 * @param      outputs   The output
 * @param      goutputs  The goutputs
 */
void split_kernel_exec(cv::Mat& input, cv::cuda::GpuMat& ginput, std::vector<cv::Mat>& outputs, std::vector<cv::cuda::GpuMat>& goutputs) {
	// upload
	cv::cuda::GpuMat goutput_collection;
	upload_outputs(outputs, goutputs, goutput_collection);
	ginput.upload(input);

  	// execute 1
  	for(int i = 0; i < goutputs.size(); ++i) {
    	cv::cuda::multiply(goutputs[i], cv::Scalar(0.5,0.5,0.5), goutputs[i]);
  	}

  	// execute 2
  	call_split_kernel<char3>(goutput_collection, ginput);    
  
  	// download
  	for (int i = 0; i < goutputs.size(); ++i) {
  		goutputs[i].download(outputs[i]);
  	}
}

/**
 * @brief      Terminates the invert kernel
 *
 *             Performs the required memory cleanup on the CPU and GPU.
 *
 * @param      input     The inputs
 * @param      ginput    The ginput
 * @param      outputs   The output
 * @param      goutputs  The goutputs
 */
void split_kernel_exit(cv::Mat& input, cv::cuda::GpuMat& ginput, std::vector<cv::Mat>& outputs, std::vector<cv::cuda::GpuMat>& goutputs) {
	for (int i = 0; i < goutputs.size(); ++i) {
		goutputs[i].release();
	}
	goutputs.clear();

	ginput.release();
}


// ----------------------------------------------------------------------


int main() {
	// Create input and output images
	std::string imagePath = "../data/image.jpg";
	cv::Mat input = cv::imread(imagePath,CV_LOAD_IMAGE_COLOR);

	if(input.empty())	{
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}

	std::vector<cv::Mat> outputs;
	outputs.push_back(cv::Mat::zeros(input.size(), input.type()));
	outputs[0].setTo(cv::Scalar(255,255,255));
	outputs.push_back(cv::Mat::zeros(input.size(), input.type()));
	outputs[1].setTo(cv::Scalar(255,255,255));

	// Create input and output gpu images
	cv::cuda::GpuMat ginput;
	std::vector<cv::cuda::GpuMat> goutputs;

	// Call the wrapper function
	split_kernel_init(input, ginput, outputs, goutputs);
	split_kernel_exec(input, ginput, outputs, goutputs);
	split_kernel_exit(input, ginput, outputs, goutputs);

	// Show the input and output
	cv::imshow("input",input);
	cv::imshow("output1",outputs[0]);
	cv::imshow("output2",outputs[1]);
	
	// Wait for key press
	cv::waitKey();

	return 0;
}
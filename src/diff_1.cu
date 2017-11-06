/**
 * @brief      Image differencing.
 *
 *             In this example we start uploading multiple input matrices to the
 *             GPU using all of the techniques from the previous examples. It is
 *             very much similar to the previous example.
 *
 *             From: https://stackoverflow.com/a/35621962/6207953
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
cv::cuda::GpuMat ginput1, ginput2;
cv::cuda::GpuMat goutput;

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
    out.x = val;
    out.y = val;
    out.z = val;
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
	uchar3 out;
  out.x = in1.x - in2.x;
  out.y = in1.y - in2.y;
  out.z = in1.z - in2.z;
  return out;
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
	uchar out;
  out = in1 - in2;
  return out;
}

/**
 * @brief      The difference kernel.
 *
 *             This implementation uses some higher level open cv structures,
 *             based on:
 *
 *             https://stackoverflow.com/a/35621962/6207953
 *
 * @param[in]  input1  The input 1
 * @param[in]  input2  The input 2
 * @param[in]  output  The output
 *
 * @tparam     T_in    The input type
 * @tparam     T_out   The output type
 */
template <typename T_in, typename T_out>
__global__ void diff_kernel(const cv::cuda::PtrStepSz<T_in> input1, const cv::cuda::PtrStepSz<T_in> input2, cv::cuda::PtrStepSz<T_out> output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= input1.cols || y >= input1.rows)
    return;

  if (x >= input2.cols || y >= input2.rows)
    return;

  T_out result;
  result = subtract_value(input2(y,x), input1(y,x));

  output(y, x) = result;
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
	ginput1.create(input1.rows, input1.cols, input1.type());
	ginput2.create(input2.rows, input2.cols, input2.type());
  	goutput.create(output.rows, output.cols, output.type());
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
	ginput1.release();
	ginput2.release();
	goutput.release();
}

/**
 * @brief      Calls the difference kernel.
 *
 * @param[in]  input1  The input 1
 * @param[in]  input2  The input 2
 * @param[in]  output  The output
 */
void call_diff_kernel(const cv::cuda::GpuMat& input1, const cv::cuda::GpuMat& input2, cv::cuda::GpuMat& output) {
	// Assert
	CV_Assert(input1.channels() == input2.channels()); 
	CV_Assert(input1.channels() == 3); 
	CV_Assert(input2.channels() == 3); 

  	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid(cv::cuda::device::divUp(input1.cols, block.x), cv::cuda::device::divUp(input1.rows, block.y));

	// Launch kernel
  	diff_kernel<uchar3, uchar3><<<grid, block>>>(input1, input2, output);

  	// Get last error
  	cudaSafeCall(cudaGetLastError());
}

/**
 * @brief      Wrapper for the difference kernel.
 *
 *             This implementation uses some higher level open cv structures,
 *             based on:
 *
 *             https://stackoverflow.com/a/35621962/6207953
 *
 * @param[in]  input1  The input 1
 * @param[in]  input2  The input 2
 * @param[in]  output  The output
 */
void diff_kernel(const cv::Mat& input1, const cv::Mat& input2, cv::Mat& output) {
  ginput1.upload(input1);
  ginput2.upload(input2);
  call_diff_kernel(ginput1, ginput2, goutput);    
  goutput.download(output);
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
	diff_kernel(input1, input2, output);
	diff_kernel_exit(input1, input2, output);

	// Show the input and output
	cv::imshow("Input1",input1);
	cv::imshow("Input2",input2);
	cv::imshow("Output",output);
	
	// Wait for key press
	cv::waitKey();

	return 0;
}
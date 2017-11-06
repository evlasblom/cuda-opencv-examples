/**
 * @brief      Image inversion.
 *
 *             This example provides a kernel that inverts the input image.
 *             However, this time it is performed by making use of readily
 *             available containers from Open CV. By starting with the low-level
 *             implementations from the previous example, we can now better
 *             understand what the available GpuMat object from Open CV exactly
 *             does: constructing and destructing a GpuMat allocates and frees
 *             GPU memory, whereas uploading and downloading perform the copying
 *             of the memory.
 *
 *             In addition, an Open CV object is used to make it easier to
 *             iterate over the raw image data, independent of the underlying
 *             data type and dimensions. To support both single-channel black
 *             and white images as well as color images, the kernel is turned
 *             into a template. Basic operations are provided for different
 *             underlying data types.
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


// global variables
cv::cuda::GpuMat ginput, goutput;


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
 * @brief      The invert kernel.
 *
 *             This implementation uses some higher level open cv structures,
 *             based on:
 *
 *             https://stackoverflow.com/a/35621962/6207953
 *
 * @param[in]  input   The input
 * @param[in]  output  The output
 *
 * @tparam     T_in    The input type
 * @tparam     T_out   The output type
 */
template <typename T_in, typename T_out>
__global__ void invert_kernel_2(const cv::cuda::PtrStepSz<T_in> input, cv::cuda::PtrStepSz<T_out> output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= input.cols || y >= input.rows)
      return;

  T_out value;
  set_value(255, value);

  T_out result;
  result = subtract_value(value, input(y,x));

  output(y, x) = result;

}

/**
 * @brief      Initializes the invert kernel
 *
 *             Performs the required memory allocation on the CPU and GPU.
 *
 * @param[in]  input   The input
 * @param      output  The output
 */
void invert_kernel_2_init(const cv::Mat& input, cv::Mat& output) {
	ginput.create(input.rows, input.cols, input.type());
    goutput.create(output.rows, output.cols, output.type());
}

/**
 * @brief      Terminates the invert kernel
 *
 *             Performs the required memory cleanup on the CPU and GPU.
 *
 * @param[in]  input   The input
 * @param      output  The output
 */
void invert_kernel_2_exit(const cv::Mat& input, cv::Mat& output) {
	ginput.release();
	goutput.release();
}

/**
 * @brief      Calls the invert kernel.
 *
 * @param[in]  input   The input
 * @param[in]  output  The output
 */
void call_invert_kernel_2(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output) {
	// Assert
	CV_Assert(input.channels() == 1 || input.channels() == 3); 

    // Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid(cv::cuda::device::divUp(input.cols, block.x), cv::cuda::device::divUp(input.rows, block.y));

	// Launch kernel
	if (input.channels() == 1) {
	  invert_kernel_2<uchar, uchar><<<grid, block>>>(input, output);
	}
	else if (input.channels() == 3) {
	  invert_kernel_2<uchar3, uchar3><<<grid, block>>>(input, output);
	}

  // Get last error
  cudaSafeCall(cudaGetLastError() );
}

/**
 * @brief      Wrapper for the invert kernel.
 *
 *             This implementation uses some higher level open cv structures,
 *             based on:
 *
 *             https://stackoverflow.com/a/35621962/6207953
 *
 * @param[in]  input   The input
 * @param[in]  output  The output
 */
void invert_kernel_2(const cv::Mat& input, cv::Mat& output) {
  ginput.upload(input);
  call_invert_kernel_2(ginput, goutput);    
  goutput.download(output);
}


// ----------------------------------------------------------------------


int main() {
	// Read input image from the disk
	std::string imagePath = "../data/image.jpg";
	cv::Mat input = cv::imread(imagePath,CV_LOAD_IMAGE_COLOR);

	if(input.empty())	{
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}

	// Create output image
	cv::Mat output = cv::Mat::zeros(input.rows,input.cols,CV_8UC3);

	// Call the wrapper function
	invert_kernel_2_init(input, output);
	invert_kernel_2(input, output);
	invert_kernel_2_exit(input, output);

	// Show the input and output
	cv::imshow("Input",input);
	cv::imshow("Output",output);
	
	// Wait for key press
	cv::waitKey();

	return 0;
}
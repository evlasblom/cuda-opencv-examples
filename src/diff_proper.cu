/**
 * @brief      Image differencing.
 *
 *             Here we reintroduce the kernel as a template. Basic operations
 *             are specified for each of the most common underlying data types
 *             of Open CV images. The corresponding operation is executed by
 *             means of function overloading. Combined, all of these techniques
 *             provide a very flexible kernel that can easily be used for
 *             different types of matrices as well as for a variable number of
 *             inputs.
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

  output(y, x) = subtract_value(inputs[1](y, x), inputs[0](y, x));
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
void upload_inputs(const std::vector<cv::Mat>& inputs, std::vector<cv::cuda::GpuMat>& ginputs, cv::cuda::GpuMat& ginput_collection) {
  if (ginputs.empty()) return;

  cv::Mat ginput_collectionCPU(1, static_cast<int>(ginputs.size()), CV_8UC(sizeof(cv::cuda::PtrStepSzb)));
  cv::cuda::PtrStepSzb* ginput_collectionCPU_ptr = ginput_collectionCPU.ptr<cv::cuda::PtrStepSzb>();

  for (int i = 0, size = ginputs.size(); i < size; ++i, ++ginput_collectionCPU_ptr) {
    *ginput_collectionCPU_ptr = ginputs[i];
    ginputs[i].upload(inputs[i]);
  }

  ginput_collection.upload(ginput_collectionCPU);
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
 * @param      inputs  The inputs
 * @param      output  The output
 */
void diff_kernel_init(std::vector<cv::Mat>& inputs, std::vector<cv::cuda::GpuMat>& ginputs, cv::Mat& output, cv::cuda::GpuMat& goutput) {
	ginputs.reserve(inputs.size());
	for (int i = 0; i < inputs.size(); ++i)	{
		ginputs.push_back(cv::cuda::GpuMat(inputs[i].rows, inputs[i].cols, inputs[i].type()));
	}

  	goutput.create(output.rows, output.cols, output.type());
}

/**
 * @brief      Wrapper for the difference kernel.
 *
 * @param      inputs  The inputs
 * @param      output  The output
 */
void diff_kernel_exec(std::vector<cv::Mat>& inputs, std::vector<cv::cuda::GpuMat>& ginputs, cv::Mat& output, cv::cuda::GpuMat& goutput) {
	// upload
	cv::cuda::GpuMat ginput_collection;
	upload_inputs(inputs, ginputs, ginput_collection);

  	// execute
  	call_diff_kernel<char3>(ginput_collection, goutput);    
  
  	// download
  	goutput.download(output);
}

/**
 * @brief      Terminates the difference kernel
 *
 *             Performs the required memory cleanup on the CPU and GPU.
 *
 * @param      inputs  The inputs
 * @param      output  The output
 */
void diff_kernel_exit(std::vector<cv::Mat>& inputs, std::vector<cv::cuda::GpuMat>& ginputs, cv::Mat& output, cv::cuda::GpuMat& goutput) {
	for (int i = 0; i < ginputs.size(); ++i) {
		ginputs[i].release();
	}
	ginputs.clear();

	goutput.release();
}


// ----------------------------------------------------------------------


int main() {
	// Create input and output images
	std::string imagePath = "../data/image.jpg";
	cv::Mat input1 = cv::imread(imagePath,CV_LOAD_IMAGE_COLOR);

	if(input1.empty())	{
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}

	cv::Mat input2 = cv::Mat::zeros(input1.size(), input1.type());
  	input1(cv::Rect(0,10, input1.cols,input1.rows-10)).copyTo(input2(cv::Rect(0,0,input1.cols,input1.rows-10)));

  	std::vector<cv::Mat> inputs;
  	inputs.push_back(input1);
  	inputs.push_back(input2);

	cv::Mat output = cv::Mat::zeros(input1.size(), input1.type());

	// Create input and output gpu images
	std::vector<cv::cuda::GpuMat> ginputs;
	cv::cuda::GpuMat goutput;

	// Call the wrapper function
	diff_kernel_init(inputs, ginputs, output, goutput);
	diff_kernel_exec(inputs, ginputs, output, goutput);
	diff_kernel_exit(inputs, ginputs, output, goutput);

	// Show the input and output
	cv::imshow("Input1",input1);
	cv::imshow("Input2",input2);
	cv::imshow("Output",output);
	
	// Wait for key press
	cv::waitKey();

	return 0;
}
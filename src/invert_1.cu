/**
 * @brief      Image inversion.
 *
 *             This example provides a kernel that inverts the input image. This
 *             is again performed by uploading the raw data from the Mat
 *             container to the GPU and iterating over all of the pixels in
 *             simultaneous threads, taking into account the number of channels
 *             using the step size. We have split the code into separate
 *             functions for convenience.
 *
 *             From:
 *             http://www.programmerfish.com/how-to-write-a-custom-cuda-kernel-with-opencv-as-host-library/
 */
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>  
#include <opencv2/core/cuda/common.hpp>


/// global variables
unsigned char *b_input, *b_output;

/**
 * @brief      The invert kernel
 *
 *             This implementation uses low level manipulation, based on:
 *
 *             http://www.programmerfish.com/how-to-write-a-custom-cuda-kernel-with-opencv-as-host-library/
 *
 * @param      input    The input image
 * @param      output   The output image
 * @param[in]  width    The image width
 * @param[in]  height   The image height
 * @param[in]  inStep   The input step
 * @param[in]  outStep  The output step
 */
__global__ void invert_kernel_1(unsigned char* input, unsigned char* output, int width, int height, int inStep, int outStep) {
	// Index of current thread
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Number of channels
  const int in_c = inStep / width;
  const int out_c = outStep / width;

	// Only valid threads perform memory I/O
	if((x < width) && (y < height))	{

		// Location of pixel
		const int in_tid = y * inStep + (in_c * x);
		const int out_tid  = y * outStep + (out_c * x);

		// Invert
		for (int i = 0; i < in_c; ++i) {
			output[out_tid+i] = static_cast<unsigned char>(255 - input[in_tid+i]);
		}
	}
}

/**
 * @brief      Initializes the invert kernel
 *
 *             Performs the required memory allocation on the CPU and GPU.
 *
 * @param[in]  input   The input
 * @param      output  The output
 */
void invert_kernel_1_init(const cv::Mat& input, cv::Mat& output) {
	// Calculate total number of bytes of input and output image
	const int inBytes = input.step * input.rows;
	const int outBytes = output.step * output.rows;

	b_input = (unsigned char *)malloc(inBytes);
	b_output = (unsigned char *)malloc(outBytes);

	// Allocate device memory
	cudaSafeCall(cudaMalloc<unsigned char>(&b_input, inBytes));
	cudaSafeCall(cudaMalloc<unsigned char>(&b_output, outBytes));
}

/**
 * @brief      Terminates the invert kernel
 *
 *             Performs the required memory cleanup on the CPU and GPU.
 *
 * @param[in]  input   The input
 * @param      output  The output
 */
void invert_kernel_1_exit(const cv::Mat& input, cv::Mat& output) {
	// Free cpu memory
	// Calling this to early may cause problems if the gpu is not
	// yet finished. We may want to rely on the os doing the right
	// cleanup procedures.
	// free(b_input);
	// free(b_output);

	// Free the device memory
	cudaSafeCall(cudaFree(b_input));
	cudaSafeCall(cudaFree(b_output));
}

/**
 * @brief      Calls the invert kernel
 *
 *             This implementation uses low level manipulation, based on:
 *
 *             http://www.programmerfish.com/how-to-write-a-custom-cuda-kernel-with-opencv-as-host-library/
 *
 * @param[in]  input   The input image
 * @param      output  The output image
 */
void call_invert_kernel_1(const cv::Mat& input, cv::Mat& output, unsigned char * d_input, unsigned char * d_output) {
	// Assert
	CV_Assert(input.channels() == output.channels() );
  CV_Assert(input.size() == output.size() );

	// Calculate total number of bytes of input and output image
	const int inBytes = input.step * input.rows;
	const int outBytes = output.step * output.rows;

	// // Allocate device memory
	// unsigned char *d_input, *d_output;
	// cudaSafeCall(cudaMalloc<unsigned char>(&d_input, inBytes));
	// cudaSafeCall(cudaMalloc<unsigned char>(&d_output, outBytes));

	// Copy data from OpenCV input image to device memory
	cudaSafeCall(cudaMemcpy(d_input, input.ptr(), inBytes, cudaMemcpyHostToDevice));

	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid(cv::cuda::device::divUp(input.cols, block.x), cv::cuda::device::divUp(input.rows, block.y));

	// Launch kernel
	invert_kernel_1<<<grid,block>>>(d_input, d_output, input.cols, input.rows, input.step, output.step);

	// Synchronize to check for any kernel launch errors
	cudaSafeCall(cudaDeviceSynchronize() );

	// Copy back data from destination device meory to OpenCV output image
	cudaSafeCall(cudaMemcpy(output.ptr(), d_output, outBytes, cudaMemcpyDeviceToHost));

	// // Free the device memory
	// cudaSafeCall(cudaFree(d_input));
	// cudaSafeCall(cudaFree(d_output));
}

/**
 * @brief      Wrapper for the invert kernel.
 *
 *             This implementation uses low level manipulation, based on:
 *
 *             http://www.programmerfish.com/how-to-write-a-custom-cuda-kernel-with-opencv-as-host-library/
 *
 * @param[in]  input   The input
 * @param[in]  output  The output
 */
void invert_kernel_1(const cv::Mat& input, cv::Mat& output) {
	call_invert_kernel_1(input, output, b_input, b_output);
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
	invert_kernel_1_init(input, output);
	invert_kernel_1(input, output);
	invert_kernel_1_exit(input, output);

	// Show the input and output
	cv::imshow("Input",input);
	cv::imshow("Output",output);
	
	// Wait for key press
	cv::waitKey();

	return 0;
}
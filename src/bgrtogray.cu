/**
 * @brief      Color conversion.
 *
 *             This example contains a very basic image processing kernel,
 *             namely that of converting a color image to a black and white
 *             image. This is achieved by copying the raw data from the Open CV
 *             Mat object to the device and manipulating the pixels by iterating
 *             over them in different threads simultaneously. Here we make use
 *             of the step size to take into account the number of channels
 *             while iterating.
 *
 *             From:
 *             http://www.programmerfish.com/how-to-write-a-custom-cuda-kernel-with-opencv-as-host-library/
 */
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>


/**
 * @brief      CUDA safe call.
 *
 * @param[in]  err          The error
 * @param[in]  msg          The message
 * @param[in]  file_name    The file name
 * @param[in]  line_number  The line number
 */
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number) {
	if(err!=cudaSuccess) {
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

/// Safe call macro.
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

/**
 * @brief      BGR to Gray Kernel
 *
 *             This is a simple image processing kernel that converts color
 *             images to black and white by iterating over the individual
 *             pixels.
 *
 * @param      input           The input
 * @param      output          The output
 * @param[in]  width           The width
 * @param[in]  height          The height
 * @param[in]  colorWidthStep  The color width step
 * @param[in]  grayWidthStep   The gray width step
 */
__global__ void bgr_to_gray_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep) {
	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
		
		//Location of gray pixel in output
		const int gray_tid  = yIndex * grayWidthStep + xIndex;

		const unsigned char blue	= input[color_tid];
		const unsigned char green	= input[color_tid + 1];
		const unsigned char red		= input[color_tid + 2];

		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output) {
	// Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,colorBytes),"CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,grayBytes),"CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	// Launch the color conversion kernel
	bgr_to_gray_kernel<<<grid,block>>>(d_input,d_output,input.cols,input.rows,input.step,output.step);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
}

int main() {
	std::string imagePath = "../data/image.jpg";

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath,CV_LOAD_IMAGE_COLOR);

	if(input.empty()){
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}

	// Create output image
	cv::Mat output(input.rows,input.cols,CV_8UC1);

	// Call the wrapper function
	convert_to_gray(input,output);

	// Show the input and output
	cv::imshow("Input",input);
	cv::imshow("Output",output);
	
	// Wait for key press
	cv::waitKey();

	return 0;
}
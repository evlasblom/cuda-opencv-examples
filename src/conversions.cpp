/**
 * @brief      Open CV conversions.
 *
 *             To better understand the relation between various high-level Open
 *             CV containers, this example demonstrates several conversions
 *             from one to another.
 */
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp> 
#include <opencv2/core/cuda/common.hpp> 
                                        

/**
 * @brief      Uploads inputs.
 *
 * @param[in]  inputs             The inputs
 * @param      ginputs            The ginputs
 * @param      ginput_collection  The ginput collection
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

  // conversions
  cv::cuda::GpuMat ginput1, ginput2;
  ginput1.create(input1.rows, input1.cols, input1.type());
  ginput2.create(input2.rows, input2.cols, input2.type());
  cv::cuda::PtrStepSzb pb_ginput1 = ginput1;
  cv::cuda::PtrStepSz<char> pchar_ginput1 = ginput1;

  // conversions (crazy)
  std::vector<cv::cuda::GpuMat> ginputs;
  ginputs.push_back(ginput1);
  ginputs.push_back(ginput2);
  cv::cuda::GpuMat ginput_collection;
  upload_inputs(inputs, ginputs, ginput_collection);
  cv::cuda::PtrStepSzb pb_ginput_collection = ginput_collection;
  cv::cuda::PtrStepSzb* ppb_ginput_collection = (cv::cuda::PtrStepSzb*)pb_ginput_collection.ptr();
  cv::cuda::PtrStepSzb pb_ginput1cast = static_cast<cv::cuda::PtrStepSzb>(ginput1);

	// Show the input and output
	cv::imshow("Input1",input1);
	cv::imshow("Input2",input2);

	// Wait for key press
	cv::waitKey();

	return 0;
}
/**
 * @brief      Hello Wolrd with an empty kernel.
 *
 *             This is a basic Hello World example where we use our first
 *             kernel, ableit empty.
 *
 *             From: http://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf
 */
 
#include <iostream>
#include <string>

 
/**
 * @brief      Emtpy kernel
 */
__global__ void mykernel(void) {

}

int main(void) {
	mykernel<<<1,1>>>();
	std::cout << "Hello world!" << std::endl;
}
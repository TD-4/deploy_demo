

# x86-ubuntu-cpp-onnxruntime-gpu测试

⌚️: 2021年8月1日

📚参考

---

## 一、环境准备

### 1. opencv

opencv库编译安装, GPU版本

### 2. onnxruntime-gpu安装

**一定要做好版本匹配**， 根据官方说明选择适合自己的版本，我的cuda是10.2，所以我选择的是1.5.2

![image-20210730145552725](imgs/image-20210730145552725.png)

![image-20210730145710614](imgs/image-20210730145710614.png)

确定版本后，去GitHub上下载对应版本即可。

![image-20210730145809008](imgs/image-20210730145809008.png)

下载后解压如下：

```
root@6c14f6aeab13:~/lib/onnxruntime-linux-x64-gpu-1.5.2# ls
C_API.md  GIT_COMMIT_ID  LICENSE  Privacy.md  README.md  ThirdPartyNotices.txt  VERSION_NUMBER  include  lib


root@6c14f6aeab13:~/lib/onnxruntime-linux-x64-gpu-1.5.2# ls include/
cpu_provider_factory.h  cuda_provider_factory.h  onnxruntime_c_api.h  onnxruntime_cxx_api.h  onnxruntime_cxx_inline.h  onnxruntime_session_options_config_keys.h


root@6c14f6aeab13:~/lib/onnxruntime-linux-x64-gpu-1.5.2# ls lib/
libonnxruntime.so  libonnxruntime.so.1.5.2


root@6c14f6aeab13:~/lib/onnxruntime-linux-x64-gpu-1.5.2#

```



## 二、运行代码

### 1. 代码结构

```
--onnxruntime_cpp
----CMakeLists.txt
----main.cpp
```

上述代码需要使用`一、环境准备`的步骤大家环境，然后才能使用。

CMakeLists.txt

```
cmake_minimum_required(VERSION 3.14)
project(deploy_test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -ldl -pthread")

# find opencv
find_package(OpenCV 4.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# find onnxruntime cpu
include_directories(/root/lib/onnxruntime-linux-x64-gpu-1.5.2/include)
set(ONNXRUNTIME_LIB_PATH "/root/lib/onnxruntime-linux-x64-gpu-1.5.2/lib")
file(GLOB LIBS "${ONNXRUNTIME_LIB_PATH}/*")

add_executable(demo main.cpp)
target_link_libraries(demo ${LIBS}   ${OpenCV_LIBS})

```

main.cpp

```
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <onnxruntime_c_api.h>

#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <vector>
#include <stdlib.h> 
#include <iostream> 
#include <string>
#include <assert.h>


using namespace cv;
using namespace std;


static constexpr const int width_ = 224;
static constexpr const int height_ = 224;
static constexpr const int channel = 3;

int main()
{
	//*************************************************************************
	// initialize  enviroment...one enviroment per process
	// enviroment maintains thread pools and other state info
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

	// initialize session options if needed
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(8);

	// If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
	// session (we also need to include cuda_provider_factory.h above which defines it)
	//#include "cuda_provider_factory.h"
	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

	// Sets graph optimization level
	// Available levels are
	// ORT_DISABLE_ALL -> To disable all optimizations
	// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
	// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
	// ORT_ENABLE_ALL -> To Enable All possible opitmizations
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	//*************************************************************************
	// create session and load model into memory
	// using squeezenet version 1.3
	// URL = https://github.com/onnx/models/tree/master/squeezenet
	const char* model_path = "/root/iqa.onnx";

	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_options);

	//*************************************************************************
	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;

	// print number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
										   // Otherwise need vector<vector<>>

	printf("Number of inputs = %zu\n", num_input_nodes);

	// iterate over all input nodes
	for (int i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name = session.GetInputName(i, allocator);
		printf("Input %d : name=%s\n", i, input_name);
		input_node_names[i] = input_name;

		// print input node types
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %d : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims = tensor_info.GetShape();
		printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
		for (int j = 0; j < input_node_dims.size(); j++)
			printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
	}

	// Results should be...
	// Number of inputs = 1
	// Input 0 : name = input
	// Input 0 : type = 1
	// Input 0 : num_dims = 4
	// Input 0 : dim 0 = 1
	// Input 0 : dim 1 = 3
	// Input 0 : dim 2 = 224
	// Input 0 : dim 3 = 224

	//*************************************************************************
	// Similar operations to get output node information.
	// Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
	// OrtSessionGetOutputTypeInfo() as shown above.
	// print number of model input nodes
	size_t num_output_nodes = session.GetOutputCount();
	std::vector<const char*> output_node_names(num_input_nodes);
	std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
										   // Otherwise need vector<vector<>>

	printf("\n\nNumber of outputs = %zu\n", num_input_nodes);

	// iterate over all input nodes
	for (int i = 0; i < num_output_nodes; i++) {
		// print input node names
		char* output_name = session.GetOutputName(i, allocator);
		printf("Output %d : name=%s\n", i, output_name);
		output_node_names[i] = output_name;

		// print input node types
		Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Output %d : type=%d\n", i, type);

		// print input shapes/dims
		output_node_dims = tensor_info.GetShape();
		printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
		for (int j = 0; j < output_node_dims.size(); j++)
			printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
	}

	// Results should be...
	// Number of inputs = 1
	// Input 0 : name = input
	// Input 0 : type = 1
	// Input 0 : num_dims = 4
	// Input 0 : dim 0 = 1
	// Input 0 : dim 1 = 3
	// Input 0 : dim 2 = 224
	// Input 0 : dim 3 = 224
	//*************************************************************************
	// Score the model using sample data, and inspect values

	size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
											   // use OrtGetTensorShapeElementCount() to get official size!

	std::vector<float> input_tensor_values(input_tensor_size);
	std::vector<const char*> output_node_names_ = { "output" };

	// read image
	Mat img = imread("/root/iqa_test.bmp");

	//// BGR-RGB
	Mat dst1;
	cvtColor(img, dst1, COLOR_BGR2RGB);

	// resize image
	const int row = 224;
	const int col = 224;
	Mat dst2;
	resize(dst1, dst2, Size(224, 224));
	imwrite("dstresize.png", dst2);

	// to tensor:image = image / 255;image = image.astype(np.float32)
	dst2.convertTo(dst2, CV_32FC3, 1 / 255.0);

	// normal : MEAN = [0.485, 0.456, 0.406] STD = [0.229, 0.224, 0.225] image[0] = (image[0] - MEAN[0]) / STD[0] image[1] = (image[1] - MEAN[1]) / STD[1] image[2] = (image[2] - MEAN[2]) / STD[2]
	std::vector<float> mean_value{ 0.485, 0.456, 0.406 };
	std::vector<float> std_value{ 0.229, 0.224, 0.225 };
	cv::Mat dst3;
	std::vector<cv::Mat> bgrChannels2(3);
	cv::split(dst2, bgrChannels2);
	for (auto i = 0; i < bgrChannels2.size(); i++)
	{
		bgrChannels2[i].convertTo(bgrChannels2[i], CV_32FC1, 1.0 / std_value[i], (0.0 - mean_value[i]) / std_value[i]);
	}
	cv::merge(bgrChannels2, dst3);

	//image = image.transpose(2, 0, 1);
	cv::Mat dst4;
	std::vector<float> dst_data;
	std::vector<cv::Mat> bgrChannels(3);
	cv::split(dst3, bgrChannels);
	for (auto i = 0; i < bgrChannels.size(); i++)
	{
		std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
		dst_data.insert(dst_data.end(), data.begin(), data.end());
	}

	for (unsigned int i = 0; i < input_tensor_size; i++)
		input_tensor_values[i] = (float)i / (input_tensor_size + 1);

	// create input tensor object from data values
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

	// score model & input tensor, get back output tensor
	double timeStart = (double)getTickCount();
	for (auto i = 0; i < 1; i++) {
		auto tmp = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	}
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time ：" << nTime << "sec\n" << endl;
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

	// Get pointer to output tensor float values
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	printf("Score for class  =  %f\n", floatarr[0]);

	// initialize input data with values in [0.0, 1.0]
	for (unsigned int i = 0; i < input_tensor_size; i++)
		input_tensor_values[i] = dst_data[i];

	timeStart = (double)getTickCount();
	for (auto i = 0; i < 1; i++) {
		auto tmp = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	}
	nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time one：" << nTime << "sec\n" << endl;
	timeStart = (double)getTickCount();
	for (auto i = 0; i < 1; i++) {
		auto tmp = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	}
	nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time one：" << nTime << "sec\n" << endl;
	timeStart = (double)getTickCount();
	for (auto i = 0; i < 1; i++) {
		auto tmp = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	}
	nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time one：" << nTime << "sec\n" << endl;
	timeStart = (double)getTickCount();
	for (auto i = 0; i < 1; i++) {
		auto tmp = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	}
	nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time one：" << nTime << "sec\n" << endl;
	timeStart = (double)getTickCount();
	for (auto i = 0; i < 1; i++) {
		auto tmp = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	}
	nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time one：" << nTime << "sec\n" << endl;


	auto output_tensors1 = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	assert(output_tensors1.size() == 1 && output_tensors1.front().IsTensor());

	// Get pointer to output tensor float values
	float* floatarr1 = output_tensors1.front().GetTensorMutableData<float>();
	printf("Score for class  =  %f\n", floatarr1[0]);

	printf("Done!\n");
	return 0;
}

```



### 2. 代码执行

```
root@6c14f6aeab13:~/onnxruntime_cpp/demo_gpu/build# rm -r *
root@6c14f6aeab13:~/onnxruntime_cpp/demo_gpu/build# cmake ..
-- The C compiler identification is GNU 7.5.0
-- The CXX compiler identification is GNU 7.5.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Found CUDA: /usr/local/cuda (found suitable exact version "10.2")
-- Found OpenCV: /usr/local (found suitable version "4.5.2", minimum required is "4.4")
-- Configuring done
-- Generating done
-- Build files have been written to: /root/onnxruntime_cpp/demo_gpu/build
root@6c14f6aeab13:~/onnxruntime_cpp/demo_gpu/build# make
Scanning dependencies of target demo
[ 50%] Building CXX object CMakeFiles/demo.dir/main.cpp.o
/root/onnxruntime_cpp/demo_gpu/main.cpp: In function 'int main()':
/root/onnxruntime_cpp/demo_gpu/main.cpp:38:47: warning: ignoring return value of 'OrtStatus* OrtSessionOptionsAppendExecutionProvider_CUDA(OrtSessionOptions*, int)', declared with attribute warn_unused_result [-Wunused-result]
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~
[100%] Linking CXX executable demo
[100%] Built target demo
root@6c14f6aeab13:~/onnxruntime_cpp/demo_gpu/build# ./demo
Using Onnxruntime C++ API
Number of inputs = 1
Input 0 : name=input
Input 0 : type=1
Input 0 : num_dims=4
Input 0 : dim 0=1
Input 0 : dim 1=3
Input 0 : dim 2=224
Input 0 : dim 3=224


Number of outputs = 1
Output 0 : name=output
Output 0 : type=1
Output 0 : num_dims=0
running time ：0.626488sec

Score for class  =  0.392816
running time one：0.00550859sec

running time one：0.00549474sec

running time one：0.0054654sec

running time one：0.00545396sec

running time one：0.00545044sec

Score for class  =  0.998471
Done!

root@6c14f6aeab13:~/onnxruntime_cpp/demo_gpu/build#

```

**Time:0.00545044sec**
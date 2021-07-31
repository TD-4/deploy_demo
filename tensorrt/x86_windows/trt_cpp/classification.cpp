#include "classification.h"
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvCaffeParser.h"
#include "common.h"
#include "logging.h"
#include "trtCommon.h"
#include "gpu_allocator.h"




using namespace nvinfer1;
using namespace nvcaffeparser1;
using std::string;
using GpuMat = cuda::GpuMat;
using namespace cv;


//@brief ��������TRT�����м����͵Ķ�ռ����ָ��
template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;


//! \brief ȫ�־�̬Logger�࣬������������TRT���������м�����
static Logger gLogger;


static inline void WriteLog(const char * szLog, int nError)
{
	std::cout << szLog << nError << std::endl;
	SYSTEMTIME st;
	GetLocalTime(&st);
	FILE *fp;
	fp = fopen(".\\trtlog.txt", "at");
	fprintf(fp, "TRTLogInfo: %d:%d:%d:%d, The Err Num is %d ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, nError);
	fprintf(fp, szLog);
	fclose(fp);
	OutputDebugStringA(szLog);
}

// ************************************** InferenceEngine (Network--> Builder-->Engine�� *****************************************
/**
 * @brief ���������࣬�ӱ���õ������ļ�����TRT����ICudaEngine
*/
class InferenceEngine
{
public:
	/**
	 * @brief ����֧��caffeģ��ת��ΪTensorRt���棬caffeģ�ͱ���ָ�������
	 * 
	 * @param model_file		��������ͼ�Ľṹ���ڵ�ͱ߶�Ӧ�����Ͳ�����
	 * @param trained_file		��������ͼ��������
	 * @param output_blob		����ͼ������
	 * @param max_batch			����ͼ�������
	 * @param data_type			������ִ�о���
		see ClassificationTensorRT Classifier
	*/
	InferenceEngine(
		const string& model_file,
		const string& trained_file,
		const string& engine_file,
		const string& output_blob = "score",
		int max_batch = 32,
		nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT);


	/**
	 * @brief ����֧��onnxģ��ת��ΪTensorRt����
	 * @param �ο���һ�����캯��
	*/
	InferenceEngine(
		const string& onnxModel,
		const string& engineName,
		bool attachSoftmax,
		int maxBatch = 32,
		nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT);


	/**
	 * @brief  ����֧���������磬����TensorRt�Դ��ӿ�ʵ�ֵ����磬
			   ע��ICudaEngine*��������Ҫ���ⲿ����Ȩ��.weights
			   ���ߴӱ���õ������ļ������л��õ�ICudaEngine
	 * 
	*/		
	InferenceEngine(ICudaEngine*);


	/**
	 * @brief �����л��Ѿ����ɵ������ļ����������ȼ��������Ӧ�������ļ�
	*/
	ICudaEngine* deSerializeEngine(const std::string& engineFile);


	/**
	 * @brief �����������ͷ�ICudaEngine��Ա
	 * todo ʹ������ָ������Ա���ͣ���ȥ�������ֶ������ڴ��ͷ�
	*/
	//~InferenceEngine();


	/**
	 * @bried ��ȡICudaEngine
	*/
	ICudaEngine* Get() const
	{
		return pEngine.get();
	}

private:
	std::shared_ptr<ICudaEngine> pEngine;
};


/**
* @brief ����֧��caffeģ��ת��ΪTensorRt���棬caffeģ�ͱ���ָ�������
*
* @param model_file		��������ͼ�Ľṹ���ڵ�ͱ߶�Ӧ�����Ͳ�����
* @param trained_file		��������ͼ��������
* @param output_blob		����ͼ������
* @param max_batch			����ͼ�������
* @param data_type			������ִ�о���
see ClassificationTensorRT Classifier
*/
InferenceEngine::InferenceEngine(
	const string& model_file,	
	const string& trained_file,	
	const string& engine_file,	
	const string& output_blob,	
	int max_batch,	
	nvinfer1::DataType data_type)
{
	if (FILE *file = fopen((engine_file).c_str(), "r")) {//ֱ�ӷ����л������ļ��õ�ICudaEngine
#ifdef TRTLOG
		std::cout << "���Ѿ����ɵ������ļ��г�ʼ���ƶ�����" << std::endl;
#endif // TRTLOG
		std::cout << "Loading " << engine_file << std::endl;
		pEngine = std::shared_ptr<nvinfer1::ICudaEngine>(deSerializeEngine(engine_file), samplesCommon::InferDeleter());
#ifdef TRTLOG
		std::cout << "��ʼ���ƶ��������" << std::endl;
#endif // TRTLOG
	}
	else {//����ICudaEngine������Ϊ�����ļ�
		std::cout << "Building Cuda Engine" << std::endl;
#ifdef TRTLOG
		std::cout << "��caffeȨֵ�ļ���ģ���ļ��������ƶ�����"<<std::endl;
#endif // TRTLOG
		SampleUniquePtr<IBuilder> builder(createInferBuilder(gLogger));
		SampleUniquePtr<INetworkDefinition>network(builder->createNetwork());
		std::cout << "Parsing Caffe Model" << std::endl;
		SampleUniquePtr<ICaffeParser> parser(createCaffeParser());
#ifdef TRTLOG
		std::cout << "�������������" << std::endl;
#endif // TRTLOG
		auto blob_name_to_tensor = parser->parse(model_file.c_str(), trained_file.c_str(), *network, data_type);
#ifdef TRTLOG
		std::cout << "�������������ɹ�"<< std::endl;
#endif // TRTLOG
		//ITopKLayer* argTopN = network->addTopK(*blob_name_to_tensor->find(output_blob.c_str()), TopKOperation::kMAX, 1, 1);
		////�������и���Ҫ����Կ�������argmaxʹ��TRT�ӿ�ʵ��
		//argTopN->getOutput(0)->setName(OUTPUT_BLOB_NAME1);
		//argTopN->getOutput(1)->setName(OUTPUT_BLOB_NAME2);
		//network->markOutput(*argTopN->getOutput(0));
		//network->markOutput(*argTopN->getOutput(1));
		//specify which tensors are outputs
		auto outShape = blob_name_to_tensor->find(output_blob.c_str())->getDimensions();
		//std::cout << outShape << std::endl;
		network->markOutput(*blob_name_to_tensor->find(output_blob.c_str()));
		if (!(outShape.d[1] == 1 && outShape.d[2] == 1)) {
			auto softmax = network->addSoftMax(*network->getOutput(0));
			//// Set softmax axis to 1 since network output has shape [1, 10] in full dims mode
			network->unmarkOutput(*network->getOutput(0));
			network->markOutput(*softmax->getOutput(0));
		}
		// Build the engine
		builder->setMaxBatchSize(max_batch);
		//builder->setFp16Mode(true);
		//builder->setInt8Mode(true);
		//builder->setMaxWorkspaceSize(0.15_GiB);
		builder->setMaxWorkspaceSize(1 << 20);
		//ʹ��FP16ʱ��鲢����DLA
		auto config = std::unique_ptr<nvinfer1::IBuilderConfig, samplesCommon::InferDeleter>(builder->createBuilderConfig());
		if (data_type == nvinfer1::DataType::kHALF&&builder->platformHasFastFp16()) {
			//���ָ���뾫���ƶϣ���������DLA���٣�û��DLA������Ȼʹ��GPU�ƶ�
			//config->setMaxWorkspaceSize(0.15_GiB);
			config->setMaxWorkspaceSize(1 << 20);
			builder->setFp16Mode(true);
			samplesCommon::enableDLA(builder.get(), config.get(), 0, true);
			pEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

		}
		else {
			pEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), samplesCommon::InferDeleter());
		}
		assert(engine_&&"Failed to create inference engine.");
#ifdef TRTLOG
		std::cout << "�ɹ������ƶ�����" << std::endl;
#endif // TRTLOG
		//���������ļ�
		SampleUniquePtr<IHostMemory> serializedModel(pEngine->serialize());
		std::ofstream p(engine_file, std::ios::binary);
		p.write((const char*)serializedModel->data(), serializedModel->size());
		p.close();
#ifdef TRTLOG
		std::cout << "�ɹ������ƶ����棬�´ε��ý�ֱ�Ӵӱ����engine�ļ��õ��ƶ�����" << std::endl;
#endif // TRTLOG
		std::cout << "Save Engine as " << engine_file << std::endl;
	}
}


/**
 * @brief onnx �ĳ�ʼ�� InferenceEngine
*/
InferenceEngine::InferenceEngine(
	const string& onnxModel, 
	const string& engineName, 
	bool attachSoftmax, 
	int maxBatch,
	nvinfer1::DataType dataType)
{
	if (FILE *file = fopen((engineName).c_str(), "r")) {
		std::cout << "Loading " << engineName << std::endl;
		pEngine = std::shared_ptr<ICudaEngine>(deSerializeEngine(engineName), samplesCommon::InferDeleter());
	}
	else {
		/** 1����������
			https://cloud.tencent.com/developer/article/1800743
			IBuilder * builder = createInferBuilder(gLogger);
			nvinfer1::INetworkDefinition * network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
			auto parser = nvonnxparser::createParser(*network, gLogger);
		*/
		SampleUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
		
		/** ԭ����ע��
		�����õ�dynamic shape���ܣ������������״���㣬���������ά��ָ��Ϊ-1������ж���dynamic shape����Ҫ����profile���ο��������ĵ�
		IOptimizationProfile* profile = builder->createOptimizationProfile();
		profile->setShapeValues();
		! \brief LEDNet����ǰ�汾onnxparserֻ֧��explicitBactch����
		! \todo ����GlobalAveragePool��֧������
		*/
		SampleUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
		
		SampleUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));

		/**2������ONNX��ʽģ���ļ�
			parser->parseFromFile(onnxModel.c_str(), 2);
			for (int i = 0; i < parser->getNbErrors(); ++i) {
				std::cout << parser->getError(i)->desc() << std::endl;
			}
		*/
		std::cout << "Parsing Onnx Model ........" << std::endl;
		int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
		if (!parser->parseFromFile(onnxModel.c_str(), verbosity))
		{
			string msg("failed to parse onnx file");
			exit(EXIT_FAILURE);
		}
		// Attach a softmax layer to the end of the network.
		if (attachSoftmax) {
			auto softmax = network->addSoftMax(*network->getOutput(0));
			// Set softmax axis to 1 since network output has shape [1, 10] in full dims mode
			softmax->setAxes(1 << 1);
			network->unmarkOutput(*network->getOutput(0));
			network->markOutput(*softmax->getOutput(0));
		}
		//�����֧��kint8��֧��khalf�ͷ���false
		if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8()))
			exit(EXIT_FAILURE);

		/**3��������������
			nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
			config->setMaxWorkspaceSize(1 << 20);
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
			ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
			IExecutionContext *context = engine->createExecutionContext();
		*/
		std::cout << "Building Cuda Engine ........" << std::endl;
		builder->setMaxBatchSize(maxBatch);
		//builder->setFp16Mode(true);
		//builder->setInt8Mode(true);
		builder->setMaxWorkspaceSize(1 << 20);
		pEngine.reset(builder->buildCudaEngine(*network), samplesCommon::InferDeleter());

		/** 4�����л�Engine
			
		*/
		SampleUniquePtr<IHostMemory> serializedModel(pEngine->serialize());
		std::ofstream p(engineName, std::ios::binary);
		p.write((const char*)serializedModel->data(), serializedModel->size());
		p.close();
		std::cout << "Save Engine as " << engineName << std::endl;
	}
	//CHECK(engine_) << "��������ʧ��";
}


/**
 * @brief ����֧���������磬����TensorRt�Դ��ӿ�ʵ�ֵ����磬
			   ע��ICudaEngine*��������Ҫ���ⲿ����Ȩ��.weights
			   ���ߴӱ���õ������ļ������л��õ�ICudaEngine
*/
InferenceEngine::InferenceEngine(ICudaEngine* engine) {
	pEngine.reset(engine, samplesCommon::InferDeleter());
}

/**
 * @bried InferenceEngine::~InferenceEngine()
 * ʹ������ָ�룬��������ʡ��
 */
//{
//	engine_->destroy();
//}


/**
 * @brief �����л��Ѿ����ɵ������ļ����������ȼ��������Ӧ�������ļ�
*/
ICudaEngine* InferenceEngine::deSerializeEngine(const std::string& engineFile)
{
	std::fstream file;
	ICudaEngine* engine;
	//ָ�����з�Ϊ\n
	file.open(engineFile, std::ios::binary | std::ios::in);
	if (!file.is_open())
	{
		fprintf(stdout, "_#_DET_#_L_#_Failed to load TRT engine\n");
		std::fflush(stdout);
	}
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);
	std::unique_ptr<char[]> data(new char[length]);
	file.read(data.get(), length);

	file.close();

	SampleUniquePtr<IRuntime>runTime(createInferRuntime(gLogger));
	assert(runTime != nullptr);
	engine = runTime->deserializeCudaEngine(data.get(), length, nullptr);
	//assert(mTrtEngine != nullptr);
	return engine;
}

// ************************************** InferenceEngine END  **********************************





// ************************************** Classifier (Engine --> Runtime)*****************************************
/**
* @brief ����ִ���࣬��enging��runtime
*/
class Classifier
{
public:
	/**
	* @brief ��������ļ���ͼ����������ִ���ƶ�
	*
	* @param engine			��������ͼ�������ļ�
	* @param isOnnx			caffe������������������Ĵ���Ϊ��ά��onnx�����������������������ά����
	* @param mean_file, label_file   ͼ��Ԥ����������Ϣ
	* @param batch_size		����batchsize
	* @param input_blob, output_blob ��λ����ͼ���������
	*/
	Classifier(
		std::shared_ptr<InferenceEngine> engine,
		bool isOnnx,
		const string& mean_file,
		const string& label_file,
		GPUAllocator* allocator,
		const int& batch_size = 16,
		const string& input_blob = "data",
		const string& output_blob = "score");

	~Classifier();

	//@brief ����predict��ɷ�������ǰ���ƶϣ������ؽ��
	std::vector<std::vector<Prediction>> Classify(const std::vector<Mat>& batchImg, int N = 5);

	//@brief ����predict��ɷָ�����ǰ���ƶϣ������ؽ��
	std::vector<cv::Mat> Segment(const std::vector<Mat>& batchImg, bool probFormat = false);

	//�ж�...
	bool Judge(const std::vector<Mat>& pairImg, float T);

	//@brief ����predict��ɼ������ǰ���ƶϣ������ؽ��
	//std::vector<std::vector<Detection>> Detect(const std::vector<Mat>& batchImg);

private:

	//@brief �����Ա������Դ��ȷ�����������λ�úʹ�С
	void SetModel();

	void SetMean(const string& mean_file);

	void SetLabels(const string& label_file);

	std::vector<std::vector<float>> Predict(const std::vector<Mat>& batchImg);

	//@brief ����GpuMat������ͼ���Ƶ��Դ��ָ��λ�ã�
	void WrapInputLayer(std::vector<GpuMat>* input_channels, int idx);

	void Preprocess(const Mat& img,	std::vector<GpuMat>* input_channels);

private:
	GPUAllocator * allocator_;					// �����Դ���ָ��  GPUAllocator
	std::shared_ptr<InferenceEngine> engine_;	// Engineָ��	   InferenceEngine
	SampleUniquePtr<IExecutionContext> context_;// ִ��������ָ��  Infer1::IExectionContext
	bool isOnnx_;								// �Ƿ���onnx�ļ�  bool
	GpuMat mean_;								// ��ֵ			   cv::cuda::GpuMat
	std::vector<string> labels_;				// labels		   std::vector<string>
	DimsCHW input_dim_;							// ��������			DimsCHW
	Size input_cv_size_;						// ���������С		Size
	float* input_layer_;						// �������뻺����   float	�����GPU��
	DimsCHW output_dim_;						// ���������С		DimsCHW
	float* output_layer_;						// �������������	float	�����GPU��
	int batch_size_;							// batchsize		int
	std::string input_blob_;					// input_blob_��caffe�����Ҫ��		string
	std::string output_blob_;					// output_blob_��caffe�����Ҫ��		string
};


/**
* @brief ��������ļ���ͼ����������ִ���ƶ�
*
* @param engine			��������ͼ�������ļ�
* @param isOnnx			caffe������������������Ĵ���Ϊ��ά��onnx�����������������������ά����
* @param mean_file, label_file   ͼ��Ԥ����������Ϣ
* @param batch_size		����batchsize
* @param input_blob, output_blob ��λ����ͼ���������
*/
Classifier::Classifier(
	std::shared_ptr<InferenceEngine> engine,	
	bool isOnnx,
	const string& mean_file,
	const string& label_file,
	GPUAllocator* allocator,
	const int& batch_size,	
	const string& input_blob,
	const string& output_blob)
	:allocator_(allocator),	engine_(engine),isOnnx_(isOnnx),batch_size_(batch_size),input_blob_(input_blob),output_blob_(output_blob)
{
#ifdef TRTLOG
	std::cout << "���������Դ棬׼��ִ���ƶ�" << std::endl;
#endif // TRTLOG
	SetModel();

#ifdef TRTLOG
	std::cout << "�Դ濪�ٳɹ�" << std::endl;
	std::cout << "׼��Ԥ�����ֵ�ͱ�ǩ�ļ������ޱ�ǩ�ļ����\"0\"��ʼ����" << std::endl;
#endif // TRTLOG
	SetMean(mean_file);
	SetLabels(label_file);

#ifdef TRTLOG
	std::cout << "Ԥ�����ֵ�ͱ�ǩ׼�����" << std::endl;
#endif // TRTLOG
}

Classifier::~Classifier()
{
#ifdef TRTLOG
	std::cout << "׼���ͷ�������������ռ���Դ�" << std::endl;
#endif // TRTLOG
	assert(cudaFree(input_layer_) == cudaSuccess && "Could not free input layer");
	assert(cudaFree(output_layer_) == cudaSuccess && "Could not free output layer");
#ifdef TRTLOG
	std::cout << "������������ռ���Դ��ͷ����" << std::endl;
#endif // TRTLOG
}


/**
 @ brief adding batching, more space should be allocated
 �����Ա������Դ��ȷ�����������λ�úʹ�С
*/
void Classifier::SetModel()
{
	ICudaEngine* engine = engine_->Get();
	context_.reset(engine->createExecutionContext());
	//CHECK(context_) << "Failed to create execution context.";
	if (input_blob_ != ""&&output_blob_ != "") {//����caffeģ��������Ϣ
#ifdef TRTLOG
		std::cout << "����caffeģ��������Ϣ" << std::endl;
#endif // TRTLOG
		assert(engine->getNbBindings() == 2);
		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// note that indices are guaranteed to be less than IEngine::getNbBindings()
		int input_index{}, output_index{};
		for (int b = 0; b < engine->getNbBindings(); ++b)
		{
			if (engine->bindingIsInput(b))
				input_index = b;
			else
				output_index = b;
		}
		input_dim_ = static_cast<DimsCHW&&>(engine->getBindingDimensions(input_index));
		input_cv_size_ = Size(input_dim_.w(), input_dim_.h());
		size_t input_size = /*input_dim_.n() */batch_size_ * input_dim_.c() * input_dim_.h() * input_dim_.w() * sizeof(float);
#ifdef TRTLOG
		std::cout << "���������Դ�" << std::endl;
#endif // TRTLOG
		cudaError_t st = cudaMalloc(&input_layer_, input_size);
		assert(st == cudaSuccess && "cudaMalloc input_layer_ falied");
		//CHECK_EQ(st, cudaSuccess) << "Could not allocate input layer.";

		output_dim_ = static_cast<DimsCHW&&>(engine->getBindingDimensions(output_index));
#ifdef TRTLOG
		std::cout << "����caffeģ�������Ϣ" << std::endl;
#endif // TRTLOG
		size_t output_size = /*output_dim_.n() **/batch_size_ * output_dim_.c() * output_dim_.h() * output_dim_.w() * sizeof(float);
#ifdef TRTLOG
		std::cout << "��������Դ�" << std::endl;
#endif // TRTLOG
		st = cudaMalloc(&output_layer_, output_size);
#ifdef TRTLOG
		std::cout << "��������Դ�׼�����" << std::endl;
#endif // TRTLOG
	}
	else {
#ifdef TRTLOG
		std::cout << "����onnxģ��������Ϣ" << std::endl;
#endif // TRTLOG

		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// note that indices are guaranteed to be less than IEngine::getNbBindings()
		assert(engine->getNbBindings() == 2);
		int input_index{}, output_index{};
		for (int b = 0; b < engine->getNbBindings(); ++b)
		{
			if (engine->bindingIsInput(b))
				input_index = b;
			else
				output_index = b;
		}

		// input_layer_ ����
		/*
		input_dim_ = static_cast<DimsCHW&&>(engine->getBindingDimensions(input_index));
		input_cv_size_ = Size(input_dim_.w(), input_dim_.h());
		*/
		//ʹ������API��������ʱ���������dims3�����ﻹ�ǻᱻ���ͳ�һ��dims3������onnxģ�ͻᱻ���ͳ�dims4��
		//��Ϊonnx������������Զ��ƶ���������㣬������Ϊ��һ�δ������onnx�������ýӿ�����ģ��Ҳ�����ƶ���������㣬
		//���ִ�е��������������dims3���ֶ�������뵽dims4
		//ͬʱҲ��Ҫ�������ά�ȣ�����ʹ�����ýӿ�ʱָ������������ֲ�������ָ��Ϊdims3
		auto input_dim_4 = static_cast<DimsNCHW&&>(engine->getBindingDimensions(input_index));
		input_cv_size_ = Size(input_dim_4.w(), input_dim_4.h());
		input_dim_ = { input_dim_4.c(), input_dim_4.h(), input_dim_4.w() };
		size_t input_size = batch_size_ * input_dim_4.c() * input_dim_4.h() * input_dim_4.w() * sizeof(float);
#ifdef TRTLOG
		std::cout << "���������Դ�" << std::endl;
#endif // TRTLOG
		cudaError_t st = cudaMalloc(&input_layer_, input_size);
		assert(st == cudaSuccess && "cudaMalloc input_layer_ falied");

		// output_layer_ ����
		auto output_dim_4 = static_cast<DimsNCHW&&>(engine->getBindingDimensions(output_index));
		output_dim_ = { output_dim_4.c(), output_dim_4.h() ? output_dim_4.h() : 1, output_dim_4.w() ? output_dim_4.w() : 1 };
		size_t output_size = batch_size_ * output_dim_4.c() * output_dim_.h() * output_dim_.w() * sizeof(float);
#ifdef TRTLOG
		std::cout << "��������Դ�" << std::endl;
#endif // TRTLOG
		st = cudaMalloc(&output_layer_, output_size);
#ifdef TRTLOG
		std::cout << "��������Դ�������" << std::endl;
#endif // TRTLOG
		assert(st == cudaSuccess && "cudaMalloc output_layer_ falied");
	}
}

/**
 @ brief  ����mean
*/
void Classifier::SetMean(const string& mean_file)
{
	if (mean_file == "") {
		Scalar zero = 0.0;
		Mat host_mean = Mat(input_cv_size_, CV_32FC(input_dim_.c()), zero);
		/*uchar pixel = host_mean.at<uchar>(0,0);
		std::cout << int(pixel) << std::endl;*/
		/*cv::imshow("host_mean", host_mean);
		cvWaitKey(0);*/
		mean_.upload(host_mean);
		return;
	}
	SampleUniquePtr<ICaffeParser> parser(createCaffeParser());
	IBinaryProtoBlob* mean_blob = parser->parseBinaryProto(mean_file.c_str());
	//CHECK(mean_blob) << "Could not load mean file.";

	DimsNCHW mean_dim = mean_blob->getDimensions();
	int c = mean_dim.c();
	int h = mean_dim.h();
	int w = mean_dim.w();
	//CHECK_EQ(c, input_dim_.c())
	//<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<Mat> channels;
	float* data = (float*)mean_blob->getData();
	for (int i = 0; i < c; ++i)
	{
		/* Extract an individual channel. */
		/*Mat channel(h, w, CV_32FC1, data);
		channels.push_back(channel);*/
		channels.emplace_back(h, w, CV_32FC1, data);
		data += h * w;
	}
	data = nullptr;
	/* Merge the separate channels into a single image. */
	Mat packed_mean;
	cv::merge(channels, packed_mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */

	//Scalar channel_mean = mean(packed_mean);
	//Mat host_mean = Mat(input_cv_size_, packed_mean.type(), channel_mean);
	//mean_.upload(host_mean);
	mean_.upload(packed_mean);

	/*uchar pixel = host_mean.at<uchar>(0,0);
	std::cout << int(pixel) << std::endl;*/
	/*cv::imshow("host_mean", host_mean);
	cvWaitKey(0);*/
}

/**
 @ brief  ����labels
*/
void Classifier::SetLabels(const string& label_file)
{
	std::ifstream labels(label_file.c_str());
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));
}


bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
	return lhs.first > rhs.first;
}

/**
 * @brief ���������з�������N��÷����,�����С�������Ŀc���������СN
 */
std::vector<int> Argmax(const std::vector<float>& v, int N)
{
	std::vector<std::pair<float, int>> pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));

	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/**
 * @brief �ָ������з���ÿ��������������ڵ���������С��c*w*h�������С��w*h
  *			update ��ʹ�����ýӿ��滻
*/
std::vector<float> Argmax(const std::vector<float>& v, int w, int h, int c) {
	std::vector<float> maxIndex(w*h, 0);
	std::vector<float> maxProb(v.begin(), v.begin() + w * h);
	for (int channel = 1; channel < c; ++channel) {
		for (int i = 0; i < w*h; ++i) {
			if (v[w*h*channel + i] > maxProb[i]) {
				maxIndex[i] = channel;
				maxProb[i] = v[w*h*channel + i];
			}
		}
	}
	return maxIndex;
}

bool Classifier::Judge(const std::vector<Mat>& pairImg, float T)
{
	int batchSize = pairImg.size();
	assert(batchSize == 2, "Support single pairwise input currently");

	std::vector<std::vector<float>> output = Predict(pairImg);

	std::vector<float>embDif(output[0].size());
	for (int i = 0; i < output[0].size(); ++i) {
		embDif[i] = output[0][i] - output[1][i];
		embDif[i] *= embDif[i];
	}
	float dis = .0;
	dis = sqrtf(accumulate(embDif.begin(), embDif.end(), .0));
	return dis < T;
}


/*
Wrap the input layer of the network in separate Mat objects (one per channel).
This way we save one memcpy operation and we don't need to rely on cudaMemcpy2D.
The last preprocessing operation will write the separate channels directly to the input layer.
�������������װ�ڵ�����Mat������(ÿ��ͨ��һ��)��
�������Ǿͽ�ʡ��һ��memcpy�������������ǲ���Ҫ������cudaMemcpy2D��
���һ��Ԥ����������ѵ�����ͨ��ֱ��д������㡣
*/
void Classifier::WrapInputLayer(std::vector<GpuMat>* input_channels, int idx)
{
	int w = input_dim_.w();
	int h = input_dim_.h();
	int c = input_dim_.c();
	int data = w * h * c * idx;
	//float* input_data = input_layer_ + width * height * input_dim_.c() * idx;
	for (int i = 0; i < input_dim_.c(); ++i)
	{
		GpuMat channel(h, w, CV_32FC1, input_layer_ + data);
		input_channels->push_back(channel);
		//input_data += width * height;
		data += h * w;
	}

}

/**
 * @brief ����������ͼƬ����Ԥ������д���Դ��У�ͨ��OPENCVʵ��
*/
void Classifier::Preprocess(const Mat& host_img, std::vector<GpuMat>* input_channels)
{
	int num_channels = input_dim_.c();
#ifdef TRTLOG
	std::cout << "��⵽����ͼ��ͨ����Ϊ:" << host_img.channels() << std::endl;
	std::cout << "��⵽��������ͨ����Ϊ: " << num_channels << std::endl;
#endif // TRTLOG

	//std::vector<float> im((uchar*)(host_img.data), (uchar*)(host_img.data) + input_dim_.c()*input_dim_.w()*input_dim_.h());
	GpuMat img(host_img, allocator_);
	//GpuMat img(host_img);
	
	/* Convert the input image to the input image format of the net. */
	GpuMat sample(allocator_);
	//GpuMat sample;
	if (img.channels() == 3 && num_channels == 1)
		cuda::cvtColor(img, sample, COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels == 1)
		cuda::cvtColor(img, sample, COLOR_RGBA2GRAY);
	else if (img.channels() == 4 && num_channels == 3)
		cuda::cvtColor(img, sample, COLOR_RGBA2RGB);
	else if (img.channels() == 1 && num_channels == 3)
		cuda::cvtColor(img, sample, COLOR_GRAY2BGR);
	else
		sample = img;

	//#ifdef TRTLOG
	//	std::cout << "����ͼƬ��СΪ " << sample.size() << std::endl;
	//	std::cout << "���������СΪ " << input_cv_size_ << std::endl;
	//#endif // TRTLOG

	/* resize image */
	GpuMat sample_resized(allocator_);
	//GpuMat sample_resized;
	if (sample.size() != input_cv_size_)
		cuda::resize(sample, sample_resized, input_cv_size_);
	else
		sample_resized = sample;

	/* ��ɸ��� */
	GpuMat sample_float(allocator_);
	//GpuMat sample_float;
	//sample_resized.convertTo(sample_float, CV_32F);
	if (num_channels == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	/* ��һ��. */
	GpuMat sample_normalized(allocator_);
	//GpuMat sample_normalized;
	if (mean_.empty())
		sample_normalized = sample_float;
	else
		cuda::subtract(sample_float, mean_, sample_normalized);
	
	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the Mat
	* objects in input_channels. */
	cuda::split(sample_normalized, *input_channels);

}

/**
 * @brief ���ࡢ���ͷָ��Ԥ�����
*/
std::vector<std::vector<float>> Classifier::Predict(const std::vector<Mat>& batchImg)
{
	int batchSize = batchImg.size();

#ifdef TRTLOG
	std::cout << "�������룬д���Դ�" << std::endl;
#endif // TRTLOG
	for (int b = 0; b < batchSize; ++b) {
		std::vector<GpuMat> input_channels;
		WrapInputLayer(&input_channels, b);
		Preprocess(batchImg[b], &input_channels);
	}
#ifdef TRTLOG
	std::cout << "�������룬д���Դ����" << std::endl;
#endif // TRTLOG

	void* buffers[2] = { input_layer_, output_layer_ };
	//cudaStream_t stream;
	//cudaStreamCreate(&stream);
	if (isOnnx_)
		context_->executeV2(buffers);
		//context_->enqueue(batchSize, buffers, stream, nullptr);
		//cudaStreamSynchronize(stream);
	else
		context_->execute(batchSize, buffers);
	
#ifdef TRTLOG
	std::cout << "ǰ��ִ�����" << std::endl;
#endif // TRTLOG

	// Asynchronously copy data from host input buffers to device input buffers
	/*
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	*/
	/*
	size_t input_size = input_dim_.c() * input_dim_.h() * input_dim_.w() * batchSize;
	std::vector<float> input(input_size);
	cudaError_t st_in = cudaMemcpy(input.data(), input_layer_, input_size * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << input[0] << std::endl;
	*/
	size_t output_size = output_dim_.c() * output_dim_.h() * output_dim_.w() * batchSize;
	std::vector<std::vector<float>> batchOutputs;
	std::vector<float> output(output_size);
#ifdef TRTLOG
	std::cout << "���д��CPU" << std::endl;
#endif // TRTLOG
	cudaError_t st = cudaMemcpy(output.data(), output_layer_, output_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (st != cudaSuccess)
		throw std::runtime_error("could not copy output layer back to host");
	for (int b = 0; b < batchSize; b++) {
		batchOutputs.push_back(std::vector<float>(output.begin() + b * output_dim_.c() * output_dim_.h() * output_dim_.w(), \
			output.begin() + (b + 1) * output_dim_.c() * output_dim_.h() * output_dim_.w()));
	}
#ifdef TRTLOG
	std::cout << "�����֯��vector" << std::endl;
#endif // TRTLOG
	return batchOutputs;
}


/**
 * @brief batching inference  for ����
 */
std::vector<std::vector<Prediction>> Classifier::Classify(const std::vector<Mat>& batchImg, int N)
{
	int batchSize = batchImg.size();
#ifdef TRTLOG
	std::cout << "ִ�з���ǰ���ƶ�" << std::endl;
#endif // TRTLOG
	std::vector<std::vector<float>> output = Predict(batchImg);
#ifdef TRTLOG
	std::cout << "����ǰ���ƶ����" << std::endl;
#endif // TRTLOG
	std::vector<std::vector<Prediction>> batchPredictions;
	auto K = N > output[0].size() ? output[0].size() : N;
	bool hasLabel = !labels_.empty();
	for (int b = 0; b < batchSize; ++b) {
		std::vector<int> maxN = Argmax(output[b], K);
		std::vector<Prediction> predictions;
		for (int i = 0; i < K; ++i)
		{
			int idx = maxN[i];
			if (hasLabel)
				predictions.push_back(std::make_pair(labels_[idx], output[b][idx]));
			else
				predictions.push_back(std::make_pair(std::to_string(idx), output[b][idx]));
		}
		batchPredictions.push_back(predictions);
	}
	return batchPredictions;
}

/**
 * @brief batching inference for �ָ�
 */
std::vector<cv::Mat> Classifier::Segment(const std::vector<Mat>& batchImg, bool probFormat) {
	int batchSize = batchImg.size();
	int w = input_cv_size_.height;
	int h = input_cv_size_.width;
#ifdef TRTLOG
	WriteLog("ִ�зָ�ǰ���ƶ�\n", 0);
#endif // TRTLOG
	std::vector<std::vector<float>> output = Predict(batchImg);
#ifdef TRTLOG
	WriteLog("�ָ�ǰ���ƶ����\n", 0);
#endif // TRTLOG
	int c = output[0].size() / (w * h);
	if (probFormat) {
		std::vector<cv::Mat> batchPredictions(batchSize);
		for (int b = 0; b < batchSize; ++b) {
			////����cloneʱ���е�ͼ����͵�һ��ͼһ����ԭ�����£�
			///*vector����push_back����ʱ���ȼ����������������ʱ����ɱ���������(Ҳ���ܲ��ǳɱ����벻ͬ�汾STLʵ�����)��
			//��һ��push_backʱ
			//����ʱ���ȸ������һ��Ԫ�ص��������ڴ��У��ٽ���push_back��Ԫ�ظ��Ƶ������ӵ��ڴ��У���mat����ĸ�����ǳ����
			//���������ڴ���д���������������е�Ԫ��ʼ�պ͵�һ��Ԫ��һ��*/

			std::vector<Mat> channels;
			for (int i = 0; i < c; ++i) {
				channels.push_back(cv::Mat(w, h, CV_32F, output[b].data() + w * h * i).clone());
			}
			merge(channels, batchPredictions[b]);
		}
		return batchPredictions;
	}
	else {
		std::vector<cv::Mat> batchPredictions;
		for (int b = 0; b < batchSize; ++b) {
			batchPredictions.push_back(cv::Mat(w, h, CV_32F, Argmax(output[b], w, h, c).data()).clone());
		}
		return batchPredictions;
	}
}

// ************************************** ǰ���ƶϾ���ִ����  END*********************************





//************************************** TensorRT API�� ******************************************
/**
 * @ brief	�Զ��� ִ��������
*/
class ExecContext
{
public:
	friend ScopedContext<ExecContext>;

	static bool IsCompatible(int device)
	{
		cudaError_t st = cudaSetDevice(device);
		if (st != cudaSuccess)
			return false;

		cuda::DeviceInfo dev_info;
		if (dev_info.majorVersion() < 3)
			return false;

		return true;
	}

	ExecContext(
		std::shared_ptr<InferenceEngine> engine,
		bool isOnnx,
		const string& mean_file,
		const string& label_file,    
		int device,
		int batch_size = 16,
		const string& input_blob = "data",
		const string& output_blob = "score"): device_(device)
	{
		cudaError_t st = cudaSetDevice(device_);

		if (st != cudaSuccess)
			throw std::invalid_argument("could not set CUDA device");

		//allocator_.reset(new GPUAllocator(0.15_GiB));
		allocator_.reset(new GPUAllocator(1 << 20));
		classifier_.reset(new Classifier(engine, isOnnx, mean_file, label_file, allocator_.get(), batch_size, input_blob, output_blob));
	}

	Classifier* TensorRTClassifier()
	{
		return classifier_.get();
	}

private:
	void Activate()
	{
		/*cudaError_t st = cudaSetDevice(device_);
		if (st != cudaSuccess)
			throw std::invalid_argument("could not set CUDA device");*/
			//allocator_->reset();
	}

	void Deactivate()
	{
	}

private:
	int device_;
	std::unique_ptr<GPUAllocator> allocator_;
	std::unique_ptr<Classifier> classifier_;
};


struct classifier_ctx
{
	ContextPool<ExecContext> pool;
};


void ClassificationTensorRT::getDevices(int* num) {
	cudaError_t st = cudaGetDeviceCount(num);
	if (st != cudaSuccess)
		throw std::invalid_argument("could not list CUDA devices");
}

void ClassificationTensorRT::setDevice()
{
	cudaError_t st = cudaSetDevice(device_);
	if (st != cudaSuccess)
		throw std::invalid_argument("device is not available");
}

/**
	* @brief ʹ��caffeѵ���õ���ģ������TRT����:

	* @param model_file	ģ�������ļ�,.prototxt��ʽ
	* @param trained_file ģ�Ͳ����ļ�,.caffemodel��ʽ
	* @param mean_file	��ֵ�ļ�,.binaryproto��ʽ
	* @param label_file	��ǩ��Ϣ,.txt��ʽ���ɣ���ָ��������������
	* @param engine_file	����TRT�����ļ�����,.engine��׺��������ڸ������ļ�����ֱ�ӽ��������ļ�����Ȼ�����ǰ�����ļ�������
	* @param maxBatch		ָ������batchsize��С�ڸ���ֵ��һ��ͼҲ����ִ��ǰ���ƶϣ�ʹ��ʱbatch��õ��ڸ���ֵ�����ܻ���������
	* @param input_blob	���������
	* @param output_blob	���������
	* @param kContextsPerDevice	���ɰ������context��contextpool���ɹ�����̵߳��ã�Ĭ��1��
								ע������kֵ���ܲ�������batchsize��Դ�����ʸ�
	* @param numGpus		ָ��ʹ��ǰ��numGpus��GPU��Ĭ��ʹ�õ�һ��
	* @param data_type	�����ƶϵľ��ȣ�TRT֧�ְ뾫�ȣ�int8�ȶ��־��ȣ��ýӿ���ֻ֧���������ȺͰ뾫�ȣ���Ӧ0��1��Ĭ��Ϊ0

*/
std::shared_ptr<classifier_ctx> ClassificationTensorRT::classifier_initialize(
	std::string model_file,
	std::string trained_file,
	std::string mean_file,
	std::string label_file,
	std::string engine_file,
	int max_batch,
	std::string input_blob,
	std::string output_blob,
	int kContextsPerDevice,
	int numGpus,
	int data_type)
{
try
{
	int device_count;
	device_ = numGpus;
	cudaError_t st = cudaGetDeviceCount(&device_count);
	std::cout << "Finding " << device_count << " CUDA Device" << std::endl;
	std::cout << "Using No." << numGpus << std::endl;
	if (st != cudaSuccess)
		throw std::invalid_argument("could not list CUDA devices");
	ContextPool<ExecContext> pool;
	for (int dev = numGpus; dev <= numGpus; ++dev)
	{
		if (!ExecContext::IsCompatible(dev))
		{
			//LOG(ERROR) << "Skipping device: " << dev;
			std::cout << "Skipping device: " << dev;
			continue;
		}
		auto dt = nvinfer1::DataType(data_type);
		std::shared_ptr<InferenceEngine> engine(new InferenceEngine(model_file, trained_file, engine_file, output_blob, max_batch, dt));

		for (int i = 0; i < kContextsPerDevice; ++i)
		{
			std::unique_ptr<ExecContext> context(new ExecContext(engine, false, mean_file, label_file, dev, max_batch, input_blob, output_blob));
			//! unique_ptr<ExecContext>���ܸ��ƣ�ֻ���ƶ�
			pool.Push(std::move(context));
		}
	}


	if (pool.Size() == 0)
		throw std::invalid_argument("no suitable CUDA device");
	std::shared_ptr<classifier_ctx> ctx(new classifier_ctx{ std::move(pool) });
	std::cout << "CUDA NO ERROR" << std::endl;
	/* Successful CUDA calls can set errno. */
	errno = 0;
	return ctx;
}
catch (const std::invalid_argument& ex)
{
	//LOG(ERROR) << "exception: " << ex.what();
	errno = EINVAL;
	return nullptr;
}
}

/**
 * @brief ʹ��pytorch�ȿ��ѵ���õ���ģ������TRT���棬��Ҫ����ת��onnx��ʽ

 * @param onnx_file	 ģ�������ļ��Ͳ����ļ�,.onnx��ʽ
 * @param label_file	 ��ǩ��Ϣ,.txt��ʽ���ɣ���ָ��������������
 * @param engine_file	 ����TRT�����ļ�����,.engine��׺��������ڸ������ļ�����ֱ�ӽ��������ļ�����Ȼ�����ǰ�����ļ�������
 * @param attachSoftmax �Ƿ���ģ�ͺ󸽼�һ��softmax�㣬����ͷָ�ģ�Ͷ���һ��softmax�㣬��siamese����û��
 * @param maxBatch		 ָ������batchsize��С�ڸ���ֵ��һ��ͼҲ����ִ��ǰ���ƶϣ�ʹ��ʱbatch��õ��ڸ���ֵ�����ܻ���������
 * @param kContextsPerDevice	���ɶ��context������߳�ͬʱ���ã�Ĭ��1
 * @param numGpus		 ָ��ʹ�õ�numGpus��GPU����Ŵ�0��ʼ
 * @param data_type	 �����ƶϵľ��ȣ�TRT֧�ְ뾫�ȣ�int8�ȶ��־��ȣ��ýӿ���ֻ֧���������ȺͰ뾫�ȣ���Ӧ0��1��Ĭ��Ϊ0
*/
std::shared_ptr<classifier_ctx> ClassificationTensorRT::classifier_initialize(
	std::string& onnx_model
	,std::string& engine_name,
	std::string& label_file,
	bool attachSoftmax,
	int maxBatch,
	int kContextsPerDevice,
	int numGpus,
	int data_type) 
{
	try
	{
		int device_count;
		device_ = numGpus;
		cudaError_t st = cudaGetDeviceCount(&device_count);
		std::cout << "Finding " << device_count << " CUDA Device" << std::endl;
		std::cout << "Using No." << numGpus << std::endl;
		if (st != cudaSuccess)
			throw std::invalid_argument("could not list CUDA devices");
		ContextPool<ExecContext> pool;
		for (int dev = numGpus; dev <= numGpus; ++dev)
		{
			if (!ExecContext::IsCompatible(dev))
			{
				//LOG(ERROR) << "Skipping device: " << dev;
				std::cout << "Skipping device: " << dev;
				continue;
			}
			auto dt = nvinfer1::DataType(data_type);
			std::shared_ptr<InferenceEngine> engine(new InferenceEngine(onnx_model, engine_name, attachSoftmax, maxBatch, dt));

			for (int i = 0; i < kContextsPerDevice; ++i)
			{
				std::unique_ptr<ExecContext> context(new ExecContext(engine, true, "", label_file, dev, maxBatch, "", ""));
				//! unique_ptr<ExecContext>���ܸ��ƣ�ֻ���ƶ�
				pool.Push(std::move(context));
			}
		}


		if (pool.Size() == 0)
			throw std::invalid_argument("no suitable CUDA device");
		std::shared_ptr<classifier_ctx> ctx(new classifier_ctx{ std::move(pool) });
		std::cout << "CUDA NO ERROR" << std::endl;
		/* Successful CUDA calls can set errno. */
		errno = 0;
		return ctx;
	}
	catch (const std::invalid_argument& ex)
	{
		//LOG(ERROR) << "exception: " << ex.what();
		errno = EINVAL;
		return nullptr;
	}
}


/**
 * @brief          ִ�з���ĺ�����ÿһ��ͼ����һ��

 * @param ctx		��ʼ�����ɾ���ǰ���ܵ�context�غ�ִ���ƶϵõ�������
 * @param batchImg	�����ƶϵ�һ��ͼƬ��С�������ͼƬ�������ɳ�����ʼ��ʱ�õ���maxBatch
 see classifier_initialize()
*/
std::vector<std::vector<Prediction>> ClassificationTensorRT::classifier_classify(std::shared_ptr<classifier_ctx> ctx, std::vector<cv::Mat>& batchImg)
{
	setDevice();
	int batchSize = batchImg.size();
	try
	{
		if (batchImg.empty())
			throw std::invalid_argument("could not decode image");

		/* In this scope an execution context is acquired for inference and it
		* will be automatically released back to the context pool when
		* exiting this scope. */
		std::vector<std::vector<Prediction>> batchPredictions;

		{
			ScopedContext<ExecContext> context(ctx->pool);
			auto classifier = context->TensorRTClassifier();
			//clock_t classifyStart = clock();

			batchPredictions = classifier->Classify(batchImg);
		}
		errno = 0;
		return batchPredictions;
	}
	catch (const std::invalid_argument&)
	{
		errno = EINVAL;
		return{};
	}
}

std::vector<cv::Mat> ClassificationTensorRT::classifier_segment(std::shared_ptr<classifier_ctx> ctx, std::vector<cv::Mat>& batchImg,bool probFormat) {
	setDevice();
	int batchSize = batchImg.size();
	try
	{
		if (batchImg.empty())
			throw std::invalid_argument("could not decode image");

		//std::cout << "Classifying Image" << std::endl;
		/* In this scope an execution context is acquired for inference and it
		* will be automatically released back to the context pool when
		* exiting this scope. */
		std::vector<cv::Mat> batchPredictions;
		{
			ScopedContext<ExecContext> context(ctx->pool);
			auto classifier = context->TensorRTClassifier();
			//clock_t classifyStart = clock();

			batchPredictions = classifier->Segment(batchImg, probFormat);
			//cv::threshold(batchPredictions[0], batchPredictions[0], 1, 255, 0);
			/*cv::imshow("res", batchPredictions[0]);
			cvWaitKey(0);*/
			//clock_t classifyEnd = clock();
			//std::cout << "batch " << batchSize << " Inference takes " << classifyEnd - classifyStart << " ms" << std::endl;
		}




		/* Write the top N predictions. */
		/*for (int b = 0;b < batchSize;++b){
		const auto p = max_element(batchPredictions[b].begin(), batchPredictions[b].end(), [](const auto& lhs, const auto& rhs) { return lhs.second <
		.second; });
		std::cout << "\nTOP 1 Prediction \n" << p->first << " : " << std::to_string(p->second * 100) << "%\n" << std::endl;

		std::cout << "\nTOP 5 Predictions\n";
		for (size_t i = 0; i < batchPredictions[b].size(); ++i)
		{
		Prediction p = batchPredictions[b][i];
		std::cout << p.first << " : " << std::to_string(p.second * 100) << "%\n";
		}
		std::cout << std::endl;
		}*/

		errno = 0;
		std::string str = "finished";
		//return _strdup(str.c_str());
		return batchPredictions;
	}
	catch (const std::invalid_argument&)
	{
		errno = EINVAL;
		//return nullptr;
		return{};
	}
}

bool ClassificationTensorRT::classifier_judge(std::shared_ptr<classifier_ctx> ctx, std::vector<cv::Mat>& pairImg, float threshold)
{
	setDevice();
	int batchSize = pairImg.size();
	try
	{
		if (pairImg.empty())
			throw std::invalid_argument("could not decode image");
		bool isSimilar;
		{
			ScopedContext<ExecContext> context(ctx->pool);
			auto classifier = context->TensorRTClassifier();
			//clock_t classifyStart = clock();

			isSimilar = classifier->Judge(pairImg, threshold);
		}
		return isSimilar;
	}
	catch (const std::invalid_argument&)
	{
		errno = EINVAL;
		//return nullptr;
		return false;
	}
}



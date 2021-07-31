#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#if defined (TYPE_RECOGNITION_API_EXPORTS)
#define TYPE_RECOGNITION_API __declspec(dllexport)
#else 
#define TYPE_RECOGNITION_API __declspec(dllimport)
#endif

#include <stddef.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>


#ifdef __cplusplus
extern "C" {
#endif

	//���ࡢ��⡢�ָ�����ʾ

	//@brief vector<Prediction>����������ʾһ��ͼ�ķ�������Prediction��ʾ<������Ŷ�>
	typedef std::pair<std::string, float> Prediction;

	//@brief cv::Mat           ������ʾ����ͼ�ָ������������������Ŷ���Ϣ

	/**
	 * @brief vector<Detection>����������ʾһ��ͼ�ļ������bbox��ʾ������Ը��ƫ���� 
	 * @param bbox[4]			������Ը��ƫ����
	 * @param det_confidence]  �����а���Ŀ������Ŷ�
	 * @param class_id			�����ӦĿ�����
	 * @param class_confidence	����Ŀ�����Ϊ��class_id������Ŷ�
	*/
	typedef struct {
		float bbox[4];
		float det_confidence;
		float class_id;
		float class_confidence;
	} Detection;

	enum class NetWorkType : int
	{
		kLUSC = 0, //!<  ���з�������
		kLUSS = 1,  //!< ���зָ�����
		kLUSD = 2,  //!< ���м������
		kRES18 = 3, //!<  ResNet18
		kLED = 4   //!<   LEDNet
	};
	class TYPE_RECOGNITION_API ClassificationTensorRT {
	public:
		typedef struct classifier_ctx classifier_ctx;


		//@brief ��ȡ���õ�GPU����
		void getDevices(int* num);


		int device_;
		//@brief ���õ�ǰʹ�õ�GPU
		void setDevice();

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
		std::shared_ptr<classifier_ctx> classifier_initialize(
			std::string model_file,
			std::string trained_file,
			std::string mean_file,
			std::string label_file,
			std::string engine_file,
			int maxBatch = 32,
			std::string input_blob = "data",
			std::string output_blob = "score",
			int kContextsPerDevice = 1,
			int numGpus = 0,
			int data_type = 0
		);

		/**
		 * @brief ʹ��pytorch�ȿ��ѵ���õ���ģ������TRT���棬��Ҫ����ת��onnx��ʽ

		 * @param onnx_file	 ģ�������ļ��Ͳ����ļ�,.onnx��ʽ
		 * @param label_file	 ��ǩ��Ϣ,.txt��ʽ���ɣ���ָ��������������
		 * @param engine_file	 ����TRT�����ļ�����,.engine��׺��������ڸ������ļ�����ֱ�ӽ��������ļ�����Ȼ�����ǰ�����ļ�������
		 * @param attachSoftmax �Ƿ���ģ�ͺ󸽼�һ��softmax�㣬����ͷָ�ģ�Ͷ���һ��softmax�㣬��siamese����û��
		 * @param maxBatch		 ָ������batchsize��С�ڸ���ֵ��һ��ͼҲ����ִ��ǰ���ƶϣ�ʹ��ʱbatch��õ��ڸ���ֵ�����ܻ���������
		 * @param input_blob	 ���������
		 * @param output_blob	 ���������
		 * @param kContextsPerDevice	���ɶ��context������߳�ͬʱ���ã�Ĭ��1
		 * @param numGpus		 ָ��ʹ�õ�numGpus��GPU����Ŵ�0��ʼ
		 * @param data_type	 �����ƶϵľ��ȣ�TRT֧�ְ뾫�ȣ�int8�ȶ��־��ȣ��ýӿ���ֻ֧���������ȺͰ뾫�ȣ���Ӧ0��1��Ĭ��Ϊ0
		*/
		std::shared_ptr<classifier_ctx> classifier_initialize(
			std::string& onnx_file,
			std::string& engine_file,
			std::string& label_file,
			bool attachSoftmax,
			int maxBatch,
			int kContextsPerDevice,
			int numGpus,
			int data_type
		);


		/**
		 * @brief һЩ�����õ������Ԥ���е�����      
		 * @param nt			NetWorkType;ָ���������ͣ����������ο�������ʼ������
		 */
		//std::shared_ptr<classifier_ctx> classifier_initialize(NetWorkType nt, int dataType,int maxBatch = 32, int kContextsPerDevice = 1, int numGPUs = 1);


		/**
		 * @brief          ִ�з���ĺ�����ÿһ��ͼ����һ��
		      
		 * @param ctx		��ʼ�����ɾ���ǰ���ܵ�context�غ�ִ���ƶϵõ�������
		 * @param batchImg	�����ƶϵ�һ��ͼƬ��С�������ͼƬ�������ɳ�����ʼ��ʱ�õ���maxBatch
		 see classifier_initialize()
		*/
		std::vector<std::vector<Prediction>> classifier_classify(std::shared_ptr<classifier_ctx>ctx, std::vector<cv::Mat>& batchImg);


		/**
		 * @brief          ִ�зָ�ĺ�����ÿһ��ͼ����һ��
		     
		* @param ctx		��ʼ�����ɾ���ǰ���ܵ�context�غ�ִ���ƶϵõ�������
		* @param batchImg	�����ƶϵ�һ��ͼƬ��С�������ͼƬ�������ɳ�����ʼ��ʱ�õ���maxBatch
		see classifier_initialize()
		*/
		std::vector<cv::Mat> classifier_segment(std::shared_ptr<classifier_ctx> ctx, std::vector<cv::Mat>& batchImg, bool probFormat = false);



		/**
		 * @brief          ִ�м��ĺ�����ÿһ��ͼ����һ��
		      
		 * @param ctx		��ʼ�����ɾ���ǰ���ܵ�context�غ�ִ���ƶϵõ�������
		 * @param batchImg	�����ƶϵ�һ��ͼƬ��С�������ͼƬ�������ɳ�����ʼ��ʱ�õ���maxBatch
			see classifier_initialize()
		*/
		//std::vector<std::vector<Detection>> classifier_detect(classifier_ctx* ctx, std::vector<cv::Mat>& batchImg);


		/**
		 * @brief          ִ�����ƶȱȶԵĺ�����ÿһ��ͼ����һ��
		      
		 * @param ctx		��ʼ�����ɾ���ǰ���ܵ�context�غ�ִ���ƶϵõ����ƶȱȶԽ��
		 * @param pairImg	�����ƶϵĳɶ�ͼƬ��ֻ֧������ͼƬ��ͼƬ���Ⱥ�˳�����޹ص�
		 * @param threshold�������ֵ�������С�ڸ���ֵ�ĳɶ�ͼ��ᱻ�ж�Ϊ���ƣ�true��
			see classifier_initialize()
		*/
		bool classifier_judge(std::shared_ptr<classifier_ctx>ctx, std::vector<cv::Mat>& pairImg, float threshold);


		//@brief �ͷ���Դ
		//void classifier_destroy(classifier_ctx* ctx);

	};

#ifdef __cplusplus
}
#endif

#endif

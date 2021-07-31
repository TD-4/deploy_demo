#include <iostream>
#include <chrono>
#include "classification.h"



cv::Rect get_rect(cv::Mat& img, int input_dim, float bbox[4]) {
	int l, r, t, b;
	if (img.cols > img.rows) {
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (input_dim - input_dim * img.rows / img.cols) / 2;
		b = bbox[1] + bbox[3] / 2.f - (input_dim - input_dim * img.rows / img.cols) / 2;
		l = l * img.cols / input_dim;
		r = r * img.cols / input_dim;
		t = t * img.cols / input_dim;
		b = b * img.cols / input_dim;
	}
	else {
		l = bbox[0] - bbox[2] / 2.f - (input_dim - input_dim * img.cols / img.rows) / 2;
		r = bbox[0] + bbox[2] / 2.f - (input_dim - input_dim * img.cols / img.rows) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l * img.rows / input_dim;
		r = r * img.rows / input_dim;
		t = t * img.rows / input_dim;
		b = b * img.rows / input_dim;
	}
	return cv::Rect(l, t, r - l, b - t);
}


int main()
{
	std::cout << __cplusplus << std::endl;
	std::string task_type = "Classification"; // Segmentation, detection,classification
	static ClassificationTensorRT CLASSIFICATION_TENSORRT;

	// *************************************** Segmentation ONNX *****************************
	if (task_type == "Segmentation") { 
		std::cout << "Segmentation" << std::endl;

		// 读取图片
		cv::Mat img4 = cv::imread("D:\\Test_TRT\\segmentation\\4.bmp", 0);	// 图片是（256，256，1）
		cv::Mat img5 = cv::imread("D:\\Test_TRT\\segmentation\\5.bmp", 0);
		cv::Mat img6 = cv::imread("D:\\Test_TRT\\segmentation\\6.bmp", 0);
		cv::Mat img7 = cv::imread("D:\\Test_TRT\\segmentation\\7judy.bmp", 0);
		cv::Mat img8 = cv::imread("D:\\Test_TRT\\segmentation\\2020-07-21  20.00.24-SurImg-ID2-二维码-产品1 克隆体1-raw FPC左R_ 1(17).bmp", 0);
		
		std::string mean = "";
		std::string label = "";
		std::string engine = "D:\\Test_TRT\\segmentation\\tunet.engine";
		std::string onnxModel = "D:\\Test_TRT\\segmentation\\unet.onnx";
		
		std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
		int time1, time2;
		int batch_size = 5;
		time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		ctx = CLASSIFICATION_TENSORRT.classifier_initialize(onnxModel, engine, label, true, batch_size, 1, 0, 1);
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;
	
		img4.convertTo(img4, CV_32F, 1.0 / 255);
		img5.convertTo(img5, CV_32F, 1.0 / 255);
		img6.convertTo(img6, CV_32F, 1.0 / 255);
		img7.convertTo(img7, CV_32F, 1.0 / 255);
		img8.convertTo(img8, CV_32F, 1.0 / 255);
		std::vector<cv::Mat>input;
		input.push_back(img4);
		input.push_back(img5);
		input.push_back(img6);
		input.push_back(img7);


		// 显示图片
		//for each (cv::Mat img in input)
		//{
		//	cv::imshow("BacthOutput", img);
		//	cv::waitKey(0);
		//}
		

		time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::vector<cv::Mat>batchPredictions;

		batchPredictions = CLASSIFICATION_TENSORRT.classifier_segment(ctx, input);
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "Segmentation Time : " << time2 - time1 << "ms" << std::endl;
		for (int i = 0; i < batchPredictions.size(); ++i) {
			cv::imshow("BacthOutput", batchPredictions[i]);
			cv::waitKey(0);
		}
		return 0;
	}

	// *************************************** Classification ONNX *****************************
	if (task_type == "Classification") {
		std::cout << "Classification" << std::endl;

		cv::Mat img = cv::imread("D:\\Test_TRT\\classification\\DD\\blackline.bmp",cv::IMREAD_GRAYSCALE);
		std::string onnxModel = "D:\\Test_TRT\\classification\\DD\\mdd_resnet34.onnx";
		//std::string label = "D:\\Test_TRT\\classification\\VT2_LABEL.txt";
		std::string label = "";
		std::string engine = "D:\\Test_TRT\\classification\\DD\\mdd_resnet34.engine";
		std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
		
		int time1, time2;
		//这里batchsize要和pytorch中保存onnx所用的batchsize一致
		int batch_size = 1;
		int availableGpuNums;
		CLASSIFICATION_TENSORRT.getDevices(&availableGpuNums);
		std::cout << "Numbers of available GPU Device is " << availableGpuNums << std::endl;
		
		time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		ctx = CLASSIFICATION_TENSORRT.classifier_initialize(onnxModel, engine, label, true, batch_size, 1, 0, 0);
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;


		//cv::resize(img, img, cv::Size(150,150));
		cv::resize(img, img, cv::Size(224, 224));
		img.convertTo(img, CV_32FC4, 1.0 / 255, 0);
		std::vector<float> mean_value{ 0.39755441968379984 };
		std::vector<float> std_value{ 0.09066523780114362 };
		img.convertTo(img, CV_32FC1, 1.0 / std_value[0], (0.0 - mean_value[0]) / std_value[0]);


		std::vector<cv::Mat>input(batch_size, img);
		std::vector<std::vector<Prediction>>batchPredictions;
		

		time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		batchPredictions = CLASSIFICATION_TENSORRT.classifier_classify(ctx, input);
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "Prediction Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;
		
		
		const auto p = std::max_element(batchPredictions[0].begin(), batchPredictions[0].end(), [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
		std::cout << "\nTOP 1 Prediction \n" << p->first << " : " << std::to_string(p->second * 100) << "%\n" << std::endl;
		std::cout << "\nTOP 5 Predictions\n";
		for (size_t i = 0; i < batchPredictions[0].size(); ++i)
		{
			Prediction p = batchPredictions[0][i];
			std::cout << p.first << " : " << std::to_string(p.second * 100) << "%\n";
		}
		
		std::cout << std::endl;
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;
		
		//CLASSIFICATION_TENSORRT.classifier_destroy(ctx);
		//system("PAUSE");
	}
	
	// *************************************** Classification Caffe *****************************
	if (task_type == "Classification_Caffe") {
		//	std::string path = "E:\\CAFFEMODEL\\1\\test";
	//	std::vector<cv::String> file_names;
	//	cv::glob(path, file_names);
	//	std::string model = "E:\\CAFFEMODEL\\1/deploy.prototxt";
	//	std::string trained = "E:\\CAFFEMODEL\\1/_iter_50000.caffemodel";
	//	std::string mean = "E:\\CAFFEMODEL\\1/mean.binaryproto";
	//	std::string label = "E:\CAFFEMODEL\1/label.txt";
	//	std::string engine = "testClassificationTRT.engine";
	//	std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
	//	int time1, time2;
	//	int batch_size = file_names.size();
	//	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	ctx = CLASSIFICATION_TENSORRT.classifier_initialize(model, trained, mean, label, engine, batch_size, "data", "prob", 1, 1, 0);
	//	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;
	//	std::vector<cv::Mat>input;
	//	for (auto&f : file_names) {
	//		auto img = cv::imread(f, 0);
	//		input.push_back(img);
	//	}
	//	std::vector<std::vector<Prediction>>batchPredictions;
	//	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	for (int i = 0; i < 1; ++i) {
	//		batchPredictions = CLASSIFICATION_TENSORRT.classifier_classify(ctx, input);
	//	}
	//	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	for (int k = 0; k < file_names.size(); ++k) {
	//		const auto p = std::max_element(batchPredictions[k].begin(), batchPredictions[k].end(), [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
	//		std::cout << file_names[k] << std::endl;
	//		std::cout << "\nTOP 1 Prediction \n" << p->first << " : " << std::to_string(p->second * 100) << "%\n" << std::endl;
	//		std::cout << "\nTOP 5 Predictions\n";
	//		for (size_t i = 0; i < batchPredictions[k].size(); ++i)
	//		{

	//			Prediction p = batchPredictions[k][i];
	//			std::cout << p.first << " : " << std::to_string(p.second * 100) << "%\n";
	//		}
	//		std::cout << std::endl;
	//	}
	//	
	//	std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;
	//	return 0;
	//	/*cv::Mat img = cv::imread("E:\\CAFFEMODEL/1.bmp");
	//	std::string model = "E:\\CAFFEMODEL/VT2_DEPLOY.prototxt";
	//	std::string trained = "E:\\CAFFEMODEL/VT2_MODEL.caffemodel";
	//	std::string mean = "E:\\CAFFEMODEL/VT2_MEAN.binaryproto";
	//	std::string label = "E:\\CAFFEMODEL/VT2_LABEL.txt";
	//	std::string engine = "ClassificationTRT.engine";*/
	//
	//	/*cv::Mat img = cv::imread("E:\\CAFFEMODEL\\caffe模型\\测试图\\9V7N90B6C0CG_C1_1_PF_BD_PL1_TBP_X3700_Y1150_D577_G821_DeepvsExp_XFlip.bmp", 0);
	//	std::string model = "E:\\CAFFEMODEL\\caffe模型\\deploy.prototxt";
	//	std::string trained = "E:\\CAFFEMODEL\\caffe模型\\unname.caffemodel";
	//	std::string mean = "E:\\CAFFEMODEL\\caffe模型\\mean.binaryproto";
	//	std::string label = "";
	//	std::string engine = "unname.engine";*/
	//	//std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
	//	//int time1, time2;
	//	//int batch_size = 32;
	//	//time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	//ctx = CLASSIFICATION_TENSORRT.classifier_initialize(model, trained, mean, label, engine, batch_size, "data", "prob", 1, 1, 0);
	//	//time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	//std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;
	//	//std::vector<cv::Mat>input(batch_size, img);
	//	//std::vector<std::vector<Prediction>>batchPredictions;
	//	//time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	//for (int i = 0;i < 100;++i) {
	//	//	batchPredictions = CLASSIFICATION_TENSORRT.classifier_classify(ctx, input);
	//	//}
	//	//time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	//const auto p = std::max_element(batchPredictions[0].begin(), batchPredictions[0].end(), [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
	//	//std::cout << "\nTOP 1 Prediction \n" << p->first << " : " << std::to_string(p->second * 100) << "%\n" << std::endl;
	//	//std::cout << "\nTOP 5 Predictions\n";
	//	//for (size_t i = 0; i < batchPredictions[0].size(); ++i)
	//	//{
	//	//	Prediction p = batchPredictions[0][i];
	//	//	std::cout << p.first << " : " << std::to_string(p.second * 100) << "%\n";
	//	//}
	//	//std::cout << std::endl;
	//	//std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;
	//	////CLASSIFICATION_TENSORRT.classifier_destroy(ctx);
	//	////system("PAUSE");
	//	//return 0;
	}
	
	// *************************************** Segmentation Caffe *****************************
	if (task_type == "Segmentation_Caffe") {
		//	cv::Mat img4 = cv::imread("E:\\CAFFEMODEL/4.bmp", 0);
	//	cv::Mat img5 = cv::imread("E:\\CAFFEMODEL/5.bmp", 0);
	//	cv::Mat img6 = cv::imread("E:\\CAFFEMODEL/6.bmp", 0);
	//	std::string model = "E:\\CAFFEMODEL/deploy备份.prototxt";
	//	std::string trained = "E:\\CAFFEMODEL/segment.caffemodel";
	//	std::string mean = "E:\\CAFFEMODEL/mean.binaryproto";
	//	std::string label = "";
	//	std::string engine = "SegmentationTRT.engine";
	//	std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
	//	int time1, time2;
	//	int maxBatch = 32;
	//	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	ctx = CLASSIFICATION_TENSORRT.classifier_initialize(model, trained, mean, label, engine, maxBatch, "data", "score", 1, 1, 1);
	//	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;
	//	//std::vector<cv::Mat>input(batch_size, img4);
	//	std::vector<cv::Mat>input;
	//	input.push_back(img4);
	//	input.push_back(img5);
	//	input.push_back(img6);
	//	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	std::vector<cv::Mat>batchPredictions;
	//	for (int i = 0;i < 100;++i) {
	//		batchPredictions = CLASSIFICATION_TENSORRT.classifier_segment(ctx, input, true);
	//	}
	//	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	std::cout << "Segmentation Time : " << time2 - time1 << "ms" << std::endl;
	//	for (int i = 0;i < batchPredictions.size();++i) {
	//		cv::imshow("BacthOutput", batchPredictions[i]);
	//		cv::waitKey(0);
	//	}
	//	return 0;
	}

	
	// *************************************** Detection ONNX *****************************
	if (task_type == "Detection") {
	//	std::cout << "Detection" << std::endl;
	//	cv::Mat img = cv::imread("D:\\Test_TRT\\detection123.jpg");
	//	std::string model = "";
	//	std::string trained = "";
	//	std::string mean = "";
	//	std::string label = "";
	//	std::string onnxModel = "D:\\Test_TRT\\segment.onnx";
	//	std::string engine = "D:\\Test_TRT\\segment.engine";
	//	std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
	//	int time1, time2;
	//	int batch_size = 16;
	//	
	//	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	ctx = CLASSIFICATION_TENSORRT.classifier_initialize(onnxModel, engine, label, false, batch_size, 1, 0, 1);
	//	//ctx = CLASSIFICATION_TENSORRT.classifier_initialize(NetWorkType::kLUSD, 0, 32, 1, 1);
	//	//ctx = CLASSIFICATION_TENSORRT.classifier_initialize(onnxModel,0);
	//	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//	std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;

//	std::vector<cv::Mat>input(batch_size, img);
//	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
//	std::vector<std::vector<Detection>> batchPredictions;
//	for (int i = 0;i < 10;++i) {
//		batchPredictions = CLASSIFICATION_TENSORRT.classifier_detect(ctx, input);
//		if(!i)
//			for (size_t j = 0; j < batchPredictions[0].size(); j++) {
//				float *p = (float*)&batchPredictions[0][j];
//				/*for (size_t k = 0; k < 7; k++) {
//					std::cout << p[k] << ", ";
//				}
//				std::cout << std::endl;*/
//				cv::Rect r = get_rect(img, 320, batchPredictions[0][j].bbox);
//				cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
//				cv::putText(img, std::to_string((int)batchPredictions[0][j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
//			}
//	}
//	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
//	std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;
//	cv::imshow("detection", img);
//	cvWaitKey(0);
//	CLASSIFICATION_TENSORRT.classifier_destroy(ctx);
//	system("PAUSE");
//	return 0;
	}

	// ************************************** onnx相似判断  ************************************** 
	/*
	{
		cv::Mat img1 = cv::imread("E:\\CAFFEMODEL/1 (1).bmp");
		cv::Mat img2 = cv::imread("E:\\CAFFEMODEL/1 (2).bmp");
		cv::Mat img3 = cv::imread("E:\\CAFFEMODEL/1 (3).bmp");
		cv::Mat img4 = cv::imread("E:\\CAFFEMODEL/1 (4).bmp");
		std::string onnxModel = "E:\\CAFFEMODEL/embedding.onnx";
		std::string label = "";
		std::string engine1 = "siamese1.engine";
		std::string engine2 = "siamese2.engine";
		//std::string engine = "alexNet.engine";
		std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
		std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx1;
		int time1, time2;
		//这里batchsize要和pytorch中保存onnx所用的batchsize一致
		int batch_size = 128;
		int availableGpuNums;

		cv::cvtColor(img1, img1, cv::COLOR_BGR2RGB);
		cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
		cv::cvtColor(img3, img3, cv::COLOR_BGR2RGB);
		cv::cvtColor(img4, img4, cv::COLOR_BGR2RGB);
		//img1.convertTo(img1, CV_32F, 1.0 / 255);
		img1.convertTo(img1, CV_32FC3, 1.0 / 255, .0);
		img2.convertTo(img2, CV_32FC3, 1.0 / 255, .0);
		img3.convertTo(img3, CV_32FC3, 1.0 / 255, .0);
		img4.convertTo(img4, CV_32FC3, 1.0 / 255, .0);

		CLASSIFICATION_TENSORRT.getDevices(&availableGpuNums);
		std::cout << "Numbers of available GPU Device is " << availableGpuNums << std::endl;
		time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		ctx = CLASSIFICATION_TENSORRT.classifier_initialize(onnxModel,engine1,label,false, batch_size, 1, 0, 0);
		ctx1 = CLASSIFICATION_TENSORRT1.classifier_initialize(onnxModel, engine2, label, false, batch_size, 1, 1, 0);
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;
		time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		bool ngSim, okSim;
		std::vector<cv::Mat>ngPair = { img1,img2 };
		std::vector<cv::Mat>okPair = { img3,img4 };
		for (int i = 0;i < 100;++i) {
			ngSim = CLASSIFICATION_TENSORRT.classifier_judge(ctx, ngPair, 0.45);
			okSim = CLASSIFICATION_TENSORRT.classifier_judge(ctx, okPair, 0.45);
		}
		for (int i = 0; i < 100; ++i) {
			ngSim = CLASSIFICATION_TENSORRT1.classifier_judge(ctx1, ngPair, 0.45);
			okSim = CLASSIFICATION_TENSORRT1.classifier_judge(ctx1, okPair, 0.45);
		}
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "\nSimilarity judgement is: \n" << ngSim << " " << okSim << std::endl;
		std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;
		//CLASSIFICATION_TENSORRT.classifier_destroy(ctx);
		//system("PAUSE");
		return 0;
	}
	*/

	// ************************************** caffe相似判断 ************************************** 
	/*
	{
		cv::Mat img1 = cv::imread("E:\\CAFFEMODEL/1 (1).bmp");
		cv::Mat img2 = cv::imread("E:\\CAFFEMODEL/1 (2).bmp");
		cv::Mat img3 = cv::imread("E:\\CAFFEMODEL/1 (3).bmp");
		cv::Mat img4 = cv::imread("E:\\CAFFEMODEL/1 (4).bmp");
		std::string weight_file = "E:\\CAFFEMODEL\\siamese\\test\\test_models\\20200717_iter_26400.caffemodel";
		std::string model_file = "E:\\CAFFEMODEL\\siamese\\test\\deploy.prototxt";
		std::string label = "";
		std::string engine = "siamese_caffe.engine";
		std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
		int time1, time2;
		int batch_size = 2;
		int availableGpuNums;

		cv::cvtColor(img1, img1, cv::COLOR_BGR2RGB);
		cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
		cv::cvtColor(img3, img3, cv::COLOR_BGR2RGB);
		cv::cvtColor(img4, img4, cv::COLOR_BGR2RGB);

		img1.convertTo(img1, CV_32FC1, 1.0 / 256, .0);
		img2.convertTo(img2, CV_32FC1, 1.0 / 256, .0);
		img3.convertTo(img3, CV_32FC1, 1.0 / 256, .0);
		img4.convertTo(img4, CV_32FC1, 1.0 / 256, .0);

		CLASSIFICATION_TENSORRT.getDevices(&availableGpuNums);
		std::cout << "Numbers of available GPU Device is " << availableGpuNums << std::endl;
		time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		ctx = CLASSIFICATION_TENSORRT.classifier_initialize(model_file, weight_file, "", label, engine, batch_size, "data", "feat", 1, 0, 0);
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;
		time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		bool ngSim, okSim;
		std::vector<cv::Mat>ngPair = { img1,img2 };
		std::vector<cv::Mat>okPair = { img3,img4 };
		for (int i = 0; i < 100; ++i) {
			//阈值0.5不一定最优
			ngSim = CLASSIFICATION_TENSORRT.classifier_judge(ctx, ngPair, 0.5);
			okSim = CLASSIFICATION_TENSORRT.classifier_judge(ctx, okPair, 0.5);
		}
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "\nSimilarity judgement is: \n" << ngSim << " " << okSim << std::endl;
		std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;
		//CLASSIFICATION_TENSORRT.classifier_destroy(ctx);
		//system("PAUSE");
		return 0;
	}*/

}






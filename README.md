# 模型部署

⌚️:2020年9月4日

---

## 1. 文件夹介绍

- onnxruntime:此目录是onnxruntime部署代码
- tvm:此目录是tvm的部署代码
- tensorrt:此目录是tensorrt的部署代码
- onnx :存放onnx模型
- imgs:存放测试图片

## 2. 环境安装 & Demo

| ID   | Architecture | Platform               | API        | FrameWork(Hardware Acceleration) | Demo                                                         |
| ---- | ------------ | ---------------------- | ---------- | -------------------------------- | ------------------------------------------------------------ |
| 01   | x86          | Ubuntu18.04+           | C++ 11     | onnxruntime1.8.0(cpu)            | [![](imgs/github.png)](onnxruntime/x86_ubuntu/x86_ubuntu_cpp_onnxruntime1.8.1.md) |
| 02   | x86          | Ubuntu18.04+           | C++ 11     | onnxruntime1.8.0(gpu)            | [![](imgs/github.png)](onnxruntime/x86_ubuntu/x86_ubuntu_cpp_onnxruntime1.8.1_gpu.md) |
| 03   | x86          | Windows10/Ubuntu18.04+ | Python3.6+ | onnxruntime1.8.0(cpu)            | [![](imgs/github.png)](onnxruntime/x86_windows/x86_win10_python_onnxruntime1.8.1.md) |
| 04   | x86          | Windows10              | Python3.6+ | onnxruntime1.8.0(gpu)            | [![](imgs/github.png)](onnxruntime/x86_windows/x86_win10_python_onnxruntime1.8.1_gpu.md) |
| 05   | x86          | Windows10              | C++ 11     | onnxruntime1.8.0(cpu)            | [![](imgs/github.png)](onnxruntime/x86_windows/x86_win10_cpp_onnxruntime1.8.1.md) |
| 06   | x86          | Windows10              | C++ 11     | onnxruntime1.8.0(gpu)            | [![](imgs/github.png)](onnxruntime/x86_windows/x86_win10_cpp_onnxruntime1.8.1_gpu.md) |
| ==   | ===          | ========               | =====      | ============                     | ==                                                           |
| 07   | x86          | Ubuntu18.04+           | C++ 11     | tensorrt                         |                                                              |
| 08   | x86          | Windows10              | C++ 11     | tensorrt                         |                                                              |
| 09   | x86          | Windows10              | Python3.6+ | tensorrt                         |                                                              |
| 10   | x86          | Ubuntu18.04+           | Python3.6+ | tensorrt                         |                                                              |
| ==   | ===          | ========               | =====      | ============                     | ==                                                           |

## 3. 测试结果

本测试结果只能代表大概情况，如果想要精确和极致的结果，请自己实验。

### ONNXRUNTIME

**Windows**

(2060 GPU, Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz)

| Windows | C++    | Python |
| ------- | ------ | :----- |
| CPU     | 0.061s | 0.047s |
| GPU     | 0.009s | 0.011s |

**Ubuntu**

(V100 GPU,  Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz,  36 core)

| Ubuntu | C++    | Python |
| ------ | ------ | :----- |
| CPU    | 0.012s | --     |
| GPU    | 0.005s | --     |

### TENSORRT


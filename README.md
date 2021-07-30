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

| ID   | Architecture | Platform               | API        | FrameWork(Hardware Acceleration) | 代码Demo                                                     |
| ---- | ------------ | ---------------------- | ---------- | -------------------------------- | ------------------------------------------------------------ |
| 01   | x86服务器    | Ubuntu18.04+           | C++ 11     | onnxruntime1.8.0(cpu)            |                                                              |
| 02   | x86服务器    | Ubuntu18.04+           | C++ 11     | onnxruntime1.8.0(gpu)            |                                                              |
| 03   | x86服务器    | Windows10/Ubuntu18.04+ | Python3.6+ | onnxruntime1.8.0(cpu)            | [![](imgs/github.png)](onnxruntime/x86_windows/x86_win10_python_onnxruntime1.8.1.md) |
| 04   | x86服务器    | Windows10              | Python3.6+ | onnxruntime1.8.0(gpu)            | [![](imgs/github.png)](onnxruntime/x86_windows/x86_win10_python_onnxruntime1.8.1_gpu.md) |
| 05   | x86服务器    | Windows10              | C++ 11     | onnxruntime1.8.0(cpu)            | [![](imgs/github.png)](onnxruntime/x86_windows/x86_win10_cpp_onnxruntime1.8.1.md) |
| 06   | x86服务器    | Windows10              | C++ 11     | onnxruntime1.8.0(gpu)            | [![](imgs/github.png)](onnxruntime/x86_windows/x86_win10_cpp_onnxruntime1.8.1_gpu.md) |
|      |              |                        |            |                                  |                                                              |
| 07   | x86服务器    | Ubuntu18.04+           | C++ 11     | tensorrt                         |                                                              |


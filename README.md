# 模型部署

⌚️:2020年9月4日

---

## 1. 文件夹介绍

```
-onnxruntime				# 此目录是onnxruntime部署代码
-tvm								# 此目录是tvm的部署代码
-tensorrt						# 此目录是tensorrt的部署代码
-installs						# 此目录是tensorrt、tvm、onnxruntime等框架在不同设备上的安装教程


```

## 2. 环境安装 & Demo

| 序号 | 设备      | 系统         | 语言       | 框架             | 环境安装                                                 | 代码Demo                                                     |
| ---- | --------- | ------------ | ---------- | ---------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| 01   | x86服务器 | Ubuntu18.04+ | Python3.6+ | onnxruntime1.8.0 | [📎](installs/x86_ubuntu20.04_python_onnxruntime1.8.1.md) | [![](installs/imgs/github.png)](onnxruntime/x86_ubuntu/README.md) |
| 02   | x86服务器 | Ubuntu18.04+ | C++ 11     | onnxruntime1.8.0 | [📎](installs/x86_ubuntu20.04_cpp_onnxruntime1.8.1.md)    | [![](installs/imgs/github.png)](onnxruntime/x86_windows/README.md) |


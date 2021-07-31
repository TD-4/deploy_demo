

# x86-ubuntu-cpp-tensorrt-gpu测试

⌚️: 2021年8月1日

📚参考

- [**官方安装教程**](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing)

---

## 一、环境准备
### 1.安装cuda

按照TensorRT支持矩阵选择Cuda版本

### 2. 安装PyCuda

如果通过Python使用tensorRT的话，需要安装Pycuda，一般情况下只需要执行pip install pycuda就可以了。

有时安装会出现一些问题，这时就需要重新编译pycuda，参考：[Installing PyCUDA on Linux](https://wiki.tiker.net/PyCuda/Installation/Linux/)

### 3. 安装tensorRT

1、 最新版本为tensorRT7，根据系统下载适合的tensorRT版本，这里下载的是TensorRT-7.0.0.11.CentOS-7.6.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz，下载地址为：https://developer.nvidia.com/nvidia-tensorrt-7x-download



2、 解压缩

`tar xzvf TensorRT-7.0.0.11.CentOS-7.6.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz`



3、 安装TensorRT wheel 文件，根据python版本选择，这里是python3.7

`cd TensorRT-7.0.0.11/python`
`pip install tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl`



4、 安装graphsurgeon wheel文件

`cd TensorRT-7.0.0.11/python`
`pip install graphsurgeon-0.4.5-py2.py3-none-any.whl`



5、 配置环境变量

`export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64`
`export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/lib64`
`source /etc/profile`



6、有时需要设置

`export LD_LIBRARY_PATH=/home/qa/TensorRT-7.0.0.11/lib:$LD_LIBRARY_PATH
source ~/.bashrc`



7、使用pip list查看安装成功否

## 二、运行代码

### 1. 代码结构
[入门代码](https://github.com/FelixFu520/README/blob/main/infer/tensorrt/Code.md)
[参考代码-1](https://github.com/FelixFu-TD/TensorRT)


上述代码需要使用`一、环境准备`的步骤大家环境，然后才能使用。
执行结果 略

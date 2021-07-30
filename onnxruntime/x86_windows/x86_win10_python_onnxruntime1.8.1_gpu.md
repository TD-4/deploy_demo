# x86-win10-python-onnxruntime-gpu测试

⌚️: 2021年8月1日

📚参考

- [官方安装](https://www.onnxruntime.ai/docs/how-to/install.html)

---

## 一、环境准备

### 1. python解析器安装、GPU驱动和CUDA相关内容

python解析器等必要库包安装。

**一定要做好版本**[匹配](https://www.onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html#requirements)

### 2.onnxruntime安装

**首先查看可用版本**

```
pip install onnxruntime-gpu==

ERROR: Could not find a version that satisfies the requirement onnxruntime== (from versions: 0.1.4, 0.2.1, 0.3.0, 0.4.0, 0.5.0, 1.0.0, 1.1.0, 1.1.1, 1.1.2, 1.2.0, 1.3.
0, 1.4.0, 1.5.1, 1.5.2, 1.6.0, 1.7.0, 1.8.0, 1.8.1)
ERROR: No matching distribution found for onnxruntime-gpu==
```

安装**onnxruntime1.8.x**

```
pip install onnxruntime-gpu==1.8.1
```

## 二、运行代码

```
打开cmd，切换到安装的环境下

cd ./python
python iqa_demo.py

output:
(torch) F:\GitHub\deploy_demo\onnxruntime\x86_windows\python>python iqa_demo.py
Loading onnx...
start ------one
Inference time with the ONNX model: 0.7024672031402588
Inference result: 0.99800885
start ------two
Inference time with the ONNX model: 0.011020898818969727
Inference result: 0.99800885
```

```
CPU

(torch) F:\GitHub\deploy_demo\onnxruntime\x86_windows\python>python iqa_demo.py
Loading onnx...
start ------
Inference time with the ONNX model: 0.04799842834472656
Inference result: 0.99800885
```

**Time: 0.011020898818969727**

**说明**

可以看到GPU第一次执行的时间为0.7s，比CPU执行0.04s还要慢。GPU第二次执行0.01s比CPU执行0.04s快。

发生这样的原因是，GPU第一次要加载模型，消耗了一点时间（个人理解）


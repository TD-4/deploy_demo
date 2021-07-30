# x86-win10-python-onnxruntime测试

⌚️: 2021年8月1日

📚参考

- [安装环境时参考官方](https://www.onnxruntime.ai/docs/how-to/install.html)

---

## 一、环境准备

### 1. python解析器安装

python解析器等必要库包安装。

**一定要做好版本匹配**

### 2.onnxruntime安装

**首先查看可用版本**

```
pip install onnxruntime==

ERROR: Could not find a version that satisfies the requirement onnxruntime== (from versions: 0.1.4, 0.2.1, 0.3.0, 0.4.0, 0.5.0, 1.0.0, 1.1.0, 1.1.1, 1.1.2, 1.2.0, 1.3.
0, 1.4.0, 1.5.1, 1.5.2, 1.6.0, 1.7.0, 1.8.0, 1.8.1)
ERROR: No matching distribution found for onnxruntime==
```

安装**onnxruntime1.8.x**

```
pip install onnxruntime==1.8.1
```

## 二、运行代码

```
打开cmd，切换到安装的环境下

cd ./python
python iqa_demo.py

output:
(torch) F:\GitHub\deploy_demo\onnxruntime\x86_windows\python>python iqa_demo.py
Loading onnx...
start ------
Inference time with the ONNX model: 0.04799842834472656
Inference result: 0.99800885
```

**Time:0.04799842834472656**

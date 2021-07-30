# x86-win10-cpp-onnxruntime-gpu测试

⌚️: 2021年8月1日

📚参考

- [安装环境时参考官方](https://www.onnxruntime.ai/docs/how-to/install.html)

---

## 一、环境准备

### 1. VS2017安装

略

### 2. VS2017新建项目

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\11.png)

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\12.png)



### 3. 安装onnxruntime

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\13.png)

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\14.png)

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\15.png)



### 4. 配置opencv

编译安装略

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\17.png)

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\18.png)

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\19.png)

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\20.png)

![](F:\GitHub\deploy_demo\onnxruntime\x86_windows\imgs\21.png)

**一定要做好版本匹配**

## 二、运行代码

### 1. 代码结构

```
--x86_windows
----cpp
------onnxruntime_demo_gpu
--------main.cpp
--------stdafx.h
--------targetver.h
```

上述代码需要使用`一、环境准备`的步骤大家环境，然后才能使用。

### 2. 代码执行

![](imgs/23.png)
# LangChain-ChatGLM

# 准备工作
models/custom_search.py 中设置你的 RapidAPIKey = ""，申请网址 https://rapidapi.com/microsoft-azure-org-microsoft-cognitive-services/api/bing-web-search1 （接口名字可以搜索：bing-web-search1）

# 执行流程

1、将本地知识数据存入knowledge/txt目录下，运行 python document.py，生成本地知识库

2、再运行 python  chainGLM.py  在chatglm-6b推理时，增加本地知识

3、修改chainGLM.py的62行，调节数据集来源本地知识库还是网络搜索

4、如果报错内容为can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
   将报错代码self.numpy()改为self.cpu().numpy()即可

# 网页Demo

## Gradio

基于Gradio的网页Demo，您可以运行本仓库中的web_demo.py：

python web_demo.py

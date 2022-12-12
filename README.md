# ginger_rgbd_reco
适用于达闼机器人视觉识别的docker工程
+ docker镜像从yolov5的demo开始，从本地加载模型，需要准备一个训练好的模型权重文件
+ restapi中集成了一些坐标变换的代码，可以自行修改
+ 本地请求文件example_request需要一张深度图和RGB图

使用方法
+ 直接在终端使用 docker build -t IMAGE_NAME:tag . 命令构建镜像，初次构建会下载一个yolov5docker的原始镜像
+ 按照需求改动restapi，修改后重新build
+ 部署方式有三种
  1. 将镜像push到达闼的rdp平台，每次运行前启动镜像，镜像有效期5h
  2. 将镜像直接run在本地，通过webvpn接入rdp平台，webvpn给出的ip有效期未知，更换vpn ip后需要重新在rdp平台配置
  3. 将镜像运行在远程服务器，一劳永逸，方便快捷，可以自由监测docker运行状态

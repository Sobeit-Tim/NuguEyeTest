# NuguEyeTest

----
## 1. 프로젝트 개요
    누구 네모를 위한 플레이를 개발하는 것이 목적인 프로젝트
    기존 누구의 AI speaker만을 이용하는 것이 아닌 카메라와 디스플레이가 존재하는 NUGU nemo에 적용할만한 아이디어를 고안하고
    이를 실제로 구현해보고자 함.


---- 
## 2. 구성
1. [Nugu Play Builder](https://developers.nugu.co.kr/#/play/playBuilder?d=1582182375657)

2. [Google Cloud Function](https://cloud.google.com/functions) ( GoogleCloudFunction/CapInfo.js )

3. Relay server ( GoogleCloudVM/server.py )

4. Device python code ( eyeTestSSDPose.py )
    
----

###2.1 Nugu Play Builder
[플레이 빌더 tree 구조 사진]

위와 같은 구조를 띄고 있음

진행되는 방식 ( 참조 )

backend proxy server와 HTTP request response를 주고 받음.


    asdadsd
>asdasd

*asd*

**asdasd**

* 1
* 2



----
## Reference
### 1. Dlib: A Machine Learning Toolkit
http://www.jmlr.org/papers/volume10/king09a/king09a.pdf
https://github.com/davisking/dlib

facial landmark detection을 위해 Caffe로 학습한 모델

[“One Millisecond Face Alignment with an Ensemble of Regression Trees,” Vahid Kazemi and Josephine Sullivan, CVPR 2014.](http://www.nada.kth.se/~sullivan/Papers/Kazemi_cvpr14.pdf)

### 2.Single Shot MultiBox Detector 
https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
Opencv 에서 사용할 수 있게 TensorFlow와 Caffe로 학습한 모델

[“Single Shot MultiBox Detector,” Wei Liu, Dragomir Anguelov, ECCV 2016.](https://arxiv.org/pdf/1512.02325.pdf)

https://github.com/weiliu89/caffe/tree/ssd 


### 3.[OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008)
https://github.com/CMU-Perceptual-Computing-Lab/openpose

https://arxiv.org/pdf/1611.08050.pdf

현재 사용한 모델은 2016년 논문에 제시된 모델을 caffe로 학습한 모델임



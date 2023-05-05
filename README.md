# RestApp

<div align="center">
<img width="329" alt="image" src="https://github.com/kwarkmc/RestApp/blob/main/Documents/pic/Logo.png?raw=true">

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkwarkmc%2FRestApp&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

# RestApp
> **한양대학교 ERICA 전자공학부 전공학회 DEFAULT 딥러닝 세미나 결과물** <br/> **개발기간: 2022.07 ~ 2022.09**

## 세미나 멤버 소개

|      배성현       |          주상현         |       곽민창         |
| :------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | 
|   <img width="160px" src="https://avatars.githubusercontent.com/u/94032326?v=4" />    |                      <img width="160px" src="https://avatars.githubusercontent.com/u/87685922?v=4" />    |                   <img width="160px" src="https://avatars.githubusercontent.com/u/41298500?v=4"/>   |
|   [@hunction](https://github.com/hunction)   |    [@SangHyun014](https://github.com/SangHyun014)  | [@kwarkmc](https://github.com/kwarkmc)  |
| 한양대학교 ERICA 전자공학부      4학년 | 한양대학교 ERICA 전자공학부     4학년 | 한양대학교 ERICA 전자공학부     3학년 |

## 딥러닝 세미나 소개

**한양대학교 ERICA 전자공학부 전공학회 DEFAULT** 에서 학기 중에 진행된 딥러닝 스터디를 바탕으로 **더욱 심화된 DL/ML** 관련 기술 및 논문에 대해 리뷰하고 실제 모델에 적용하여 개발 경험을 쌓고자 하는 목적으로 시작된 학생 모임이다.

## 프로젝트 소개

딥러닝 세미나에서 진행한 첫 번째 프로젝트로, 학교 주변 카페와 음식점의 사진을 직접 팀원들이 사진을 찍어 DataSet을 만들었고, 1000개 이상의 이미지를 Augmentation을 통해 다시 증강하여 CNN Network Model을 이용하여 Classification을 위해 학습시켰다.

ResNet 등의 실제 모델을 사용하여 학습을 진행하고, 직접 CNN Network를 이해하기 위해 Sequential Model을 직접 구성하여 Data를 Input 하는 부분부터 Output으로 Classification 하는 일련의 과정을 이해할 수 있었다.

## 시작 가이드 🚩
### Requirements
For building and running the application you need:

- [Anaconda3](https://www.anaconda.com/download/)
- [Tensorflow 2.10.0](https://www.tensorflow.org/versions?hl=ko)
  - Windows 환경에서 개발을 진행했기 때문에 Windows에서 GPU를 지원하는 Version을 사용하였다.
- [Keras](https://www.tensorflow.org/guide/keras?hl=ko)
- [Jupyter Notebook](https://jupyter.org/)
- [Pycharm](https://www.jetbrains.com/ko-kr/pycharm/download/#section=windows)

### Installation
``` bash
$ git clone git@github.com:kwarkmc/RestApp.git
$ cd RestApp
```
#### For Jupyter Notebook
``` bash
$ notebook ./RestApp.ipynb
$ notebook ./ResNet_example.ipynb
```

#### For Pycharm or VSCode
``` bash
$ python ./RestApp.py
$ python ./ResNet_example.ipynb
```

---

## Stacks 📚

### Environment
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white)
   

### Framework
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) 
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

### Development
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

### Communication
![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)          

---
## 학습 결과 📺
| Sequential Model 결과  |  ResNet Model 결과   |
| :-------------------------------------------: | :------------: |
|  <img width="329" src="https://github.com/kwarkmc/RestApp/blob/main/Documents/pic/RestApp%20result.png?raw=true"/> |  <img width="329" src="https://github.com/kwarkmc/RestApp/blob/main/Documents/pic/ResNet_example_result.png?raw=true"/>|  
| Layer에 따른 Accuracy의 변화   |  2개 - 3개 - 4개 - 5개   |  
| <img width="329" src="https://github.com/kwarkmc/RestApp/blob/main/Documents/pic/Result_model1.JPG?raw=true"/>   |  <img width="329" src="https://github.com/kwarkmc/RestApp/blob/main/Documents/pic/Result_model2.JPG?raw=true"/>     |

> Layer가 많아지면 Overfitting이 발생하여 오히려 Accuracy가 낮게 나올 수 있다는 것을 볼 수 있다.
---
## 주요 기능 📦

### ❗ Custom DataSet을 폴더에서 Open 하여 Data와 Label로 준비
- 파일의 이름에 Labeling이 되어있고, OS API를 Import 하여 20개의 라벨에 맞춰 병합
- 학습에 사용하기 위해 `LabelBinarizer()`를 이용하여 원핫 인코딩
- train / Test / Validation 데이터를 `train_test_split()` 을 이용하여 나눠서 구현
- 학습에 대한 검증은 Validation 데이터로, 실전 정확도는 Test 데이터를 사용하여 Overfitting을 방지할 수 있다.

### ❗ 데이터 증강 테크닉 사용
- `keras.preprocessing.image` 의 `ImageDataGenerator`를 사용하여 **Rotation / Zoom / Shift / Shear / Flip** 등의 CV적 Augmentation을 진행하여 학습에 사용하였다.

### ❗ 모델을 H5 파일로 저장하여 Weight만 예측에 사용
- 학습한 Weight들을 H5 파일로 저장하여 Python 파일 외부에서 입력 데이터에 대해 예측을 진행할 때 매번 학습을 하지 않고 모델을 다시 불러와 쉽게 사용할 수 있다.
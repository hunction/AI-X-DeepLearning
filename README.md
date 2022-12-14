# AI-X-DeepLearning
### Final Project

Video : https://youtu.be/8zEq1a3dJ0M

# Title : 신입생을 위한 학교주변 카페 및 식당 분류

# Members :
          배성현 , 전자공학부 , 2017006635 , hyung50300@gmail.com
          신준하 , 전자공학부 , 2017006753 , ipip0114@naver.com
          곽민창 , 전자공학부 , 2018038640 , kwarkmc@hanyang.ac.kr
          홍노준 , 전자공학부 , 2022056108 , nojun1573@naver.com

# Index
###           1. Proposal
###           2. DataSets
###           3. Methodology
###           4. Evaluation & Analysis
###           5. Related Works
###           6. Conclusion: Discussion
          

          
#  1. Proposal ( Option A )
##        - Motivation 
신입생이 학교에 처음 왔을때 , 그들은 어느 음식점과 카페에서 무엇을 파는지 , 얼마에 제공하는지를 알수가 없습니다.
그리고 가게의 외형만 봐서는 어느 이름의 가게인지 추측하기 힘들 수 있습니다.
따라서 우리는 가게의 외형 이미지를 통해 가게를 분류하는 모델을 학습시킬 예정입니다.
( 가게의 Info를 제공하는 것은 어플리케이션의 역할이라 생각되어 어플리케이션은 제작하지 않습니다. )
- 단순 모델학습이 수업의 목적이라 생각하여 취지를 벗어난 어플리케이션은 제작하지 않습니다.
추가적으로 가게의 Info 제공은 Dictionary 형태의 구조라고 생각하기 때문에 단순 입력이라고 판단하여 구현하지 않습니다.

##  * 예시 ) 대학생 커뮤니티의 학기초 모습
![ex1](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ex1.jpg?raw=true)
                                                                                                 
실전 데이터를 통한 단순한 분류 네트워크를 생성 및 학습, 기존의 모델과의 비교를 통해 같은 팀원에게 개발 기회를 주고
추가적으로 각각의 CNN 연산자의 하이퍼 파라미터와 모델 성능의 관계를 확인하려 합니다. 
                       
##        - 입력과 출력 ( PipeLine ) 
입력으로 가게의 외관사진을 넣게 된다면 , 모델은 출력을 ' 스타벅스 ' 와 같은 예시로 분류하게 됩니다.
이미지 -> 라벨 ( 추후에 라벨을 통한 정보를 표출하기 위해 )
                                              
##        - At the end 
최종적으로 , 실제 데이터셋을 적용하여 학습시킨 CNN classifier를 기존의 모델들과 비교하여
각각의 연산자가 모델이 미치는 영향 ( 연산량 , 학습시간 , 파라미터 수 , 정확도 ) 을 확인하려 합니다.
           
       
#  2. DataSets
- Kaggle이나 다른 유명 오픈소스 데이터를 사용할 수 있지만 , 이번 프로젝트의 목적은 오픈소스가 아닌 실전 데이터의 적용이 목적이기 때문에,
자체적으로 취득한 데이터를 사용하기로 합니다.

## * 예시 ) 카페누엘의 이미지 총 18장 
## ( 전체적인 이미지의 갯수를 통일시키기 위해 학습시 3장은 제거 )          
![ex2](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ex2.PNG?raw=true)

          
( 10개의 식당 & 10개의 카페 for 15 pictures for each class ) - 총 20 * 15 의 300장의 이미지를 오리지널 데이터로 사용합니다.( 좌측 , 우측 , 정면 각 5장씩 )
각각의 이미지는 각기 다른 팀원이 찍어 해상도가 다르기 때문에 , 이미지의 해상도를 단일화 시키기 위한 전처리를 진행합니다.
( 또한 모델의 학습을 위해 직사각형 -> 정사각형 )
또한 데이터의 수가 적기 때문에 , 데이터 증강기술을 통해 학습데이터를 증강. ( 20 * 15  -> 20 * 15 * 10 ) - 이미지의 회전 또는 수평이동을 적용시킵니다.
          
따라서 총 3,000장의 이미지를 학습 데이터로 사용합니다. ( 각각의 클래스 별로 150장. 좌측사진 , 우측사진 , 정면사진 50장씩 ) 
          
#  3. Methodology
- 실제적인 네트워크 개발 경험이 목적이기 때문에 , tensorflow의 keras 모듈에서 지원하는 Conv2D / Pooling / Activation 등의 연산자를 이용한
네트워크 설계가 목적입니다.
filter size , activation function , depth 등의 하이퍼 파라미터의 조절을 통한 실험을 진행하였습니다.
그리고 이미지 분류의 대표적인 모델인 AlexNet , VGG-19 , Resnet50d 과 비교하여 성능을 비교했습니다.
          
##          Experiment Conditions ( 네트워크 구조 및 특이사항 )

###          a. AlexNet - 홍노준
AlexNet은 2012년에 열린 ILSVRC 대회에서 TOP 5 test error 15.4%를 기록해 1위를 차지한 네트워크로 CNN의 우수함을 전세계에 입증한 네트워크이다. AlexNet 네트워크이후로 CNN 구조의 GPU 구현과 dropout 적용이 보편화되었다.

AlexNet의 구조는 LeNet-5와 크게 다르지 않다. 위아래로 filter가 절반씩 나뉘어 2개의 GPU로 병렬연산을 수행하는 것이 가장 큰 특징이라고 할 수 있다. 총 8개의 레이어로 구성되어 있으며, 부분적으로 max-pooling가 적용된 Convolution layer가 5개, fully-connected layers 3개로 구성되어있다. 2, 4, 5번째 Convolution layer는 전 단계에서 같은 채널의 특성맵만 연결되어 있는 반면에, 3번째 Convolution layer는 전 단계의 두 채널의 특성 맵과 모두 연결되어있다. Input image는 RGB 이미지로 224×224×3이다.

![Alexnet1](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Alexnet1.png)

AlexNet을 자세히 살펴보면 다음과 같이 [Input layer]-[Conv1]-[MaxPool1]-[Norm1]-[Conv2]-[MaxPool2]-[Norm2]-[Conv3]-[Conv4]-[Conv5]-[MaxPool3]-[FC1]-[FC2]-[Output layer] 로 구성되어 있다. 여기서 Norm은 수렴속도를 높이기 위해 local response normalization을 하는 것으로 이 local response normalization은 특성맵의 차원을 변화시키지 않는다는 특징을 가지고 있다.

![Alexnet2](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Alexnet2.png)

전체적인 층의 구조는 다음과 같으며 Input image는 RGB 이미지로 224×224×3이다.

1. 1층 (Convolution layer) : 96개의 11 x 11 x 3 필터커널로 입력 영상을 컨볼루션해준다. 컨볼루션 stride를 4로 설정했고, zero-padding은 사용하지 않았다. zero-padding은 컨볼루션으로 인해 특성맵의 사이즈가 축소되는 것을 방지하기 위해, 또는 축소되는 정도를 줄이기 위해 영상의 가장자리 부분에 0을 추가하는 것이다. 결과적으로 55 x 55 x 96 특성맵(96장의 55 x 55 사이즈 특성맵들)이 산출된다. 그 다음에 ReLU 함수로 활성화해준다. 이어서 3 x 3 overlapping max pooling이 stride 2로 시행된다. 그 결과 27 x 27 x 96 특성맵을 갖게 된다. 그 다음에는 수렴 속도를 높이기 위해 local response normalization이 시행된다. local response normalization은 특성맵의 차원을 변화시키지 않으므로, 특성맵의 크기는 27 x 27 x 96으로 유지된다.

2. 2층 (Convolution layer) : 256개의 5 x 5 x 48 커널을 사용하여 전 단계의 특성맵을 컨볼루션해준다. stride는 1로, zero-padding은 2로 설정했다. 따라서 27 x 27 x 256 특성맵(256장의 27 x 27 사이즈 특성맵)을 얻게 된다. 역시 ReLU 함수로 활성화한다. 그 다음에 3 x 3 overlapping max pooling을 stride 2로 시행한다. 그 결과 13 x 13 x 256 특성맵을 얻게 된다. 그 후 local response normalization이 시행되고, 특성맵의 크기는 13 x 13 x 256으로 그대로 유지된다. 

3. 3층 (Convolution layer) : 384개의 3 x 3 x 256 커널을 사용하여 전 단계의 특성맵을 컨볼루션해준다. stride와 zero-padding 모두 1로 설정한다. 따라서 13 x 13 x 384 특성맵(384장의 13 x 13 사이즈 특성맵)을 얻게 된다. 역시 ReLU 함수로 활성화한다. 

4. 4층 (Convolution layer) : 384개의 3 x 3 x 192 커널을 사용해서 전 단계의 특성맵을 컨볼루션해준다. stride와 zero-padding 모두 1로 설정한다. 따라서 13 x 13 x 384 특성맵(384장의 13 x 13 사이즈 특성맵)을 얻게 된다. 역시 ReLU 함수로 활성화한다. 

5. 5층 (Convolution layer) : 256개의 3 x 3 x 192 커널을 사용해서 전 단계의 특성맵을 컨볼루션해준다. stride와 zero-padding 모두 1로 설정한다. 따라서 13 x 13 x 256 특성맵(256장의 13 x 13 사이즈 특성맵)을 얻게 된다. 역시 ReLU 함수로 활성화한다. 그 다음에 3 x 3 overlapping max pooling을 stride 2로 시행한다. 그 결과 6 x 6 x 256 특성맵을 얻게 된다. 

6. 6층 (Fully connected layer) : 6 x 6 x 256 특성맵을 flatten해서 6 x 6 x 256 = 9216차원의 벡터로 만들어준다. 그것을 여섯번째 레이어의 4096개의 뉴런과 fully connected 해준다. 그 결과를 ReLU 함수로 활성화한다. 

7. 7층 (Fully connected layer) : 4096개의 뉴런으로 구성되어 있다. 전 단계의 4096개 뉴런과 fully connected되어 있다. 출력 값은 ReLU 함수로 활성화된다. 

8. 8층 (Fully connected layer): 1000개의 뉴런으로 구성되어 있다. 전 단계의 4096개 뉴런과 fully connected되어 있다. 1000개 뉴런의 출력값에 softmax 함수를 적용해 1000개 클래스 각각에 속할 확률을 나타낸다.



###          b. VGG-19 - 배성현
![vgg](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/vgg.png?raw=true)
VGGNet은 VGG 팀에서 개발한 CNN( Convolutional Neural Network )으로써 이전에 ImageNet 대회에서 우승한
AlexNet에 기반하여 네트워크의 깊이( Depth ) 가 모델의 성능에 끼치는 영향을 실험했다는 것에 의의가 있다.
여기서는 단순히 깊이만을 실험함으로써 나머지 하이퍼파라미터 ( filter 크기 , stride 등 ) 을 고정한 채
깊이만을 늘리는 실험을 진행하였다.

결과론적으로 네트워크의 깊이가 단순한게 늘어남에 따라 모델의 성능이 어느 수준까지는 좋아짐을 확인했다.

![vgg_1](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/vgg_1.png?raw=true)

전체적인 층의 구조는 다음과 같다.
0) 인풋: 224 x 224 x 3 이미지(224 x 224 RGB 이미지)를 입력받을 수 있다. 

1) 1층(conv1_1): 64개의 3 x 3 x 3 필터커널로 입력이미지를 컨볼루션해준다. zero padding은 1만큼 해줬고, 컨볼루션 보폭(stride)는 1로 설정해준다. zero padding과 컨볼루션 stride에 대한 설정은 모든 컨볼루션층에서 모두 동일하니 다음 층부터는 설명을 생략하겠다. 결과적으로 64장의 224 x 224 특성맵(224 x 224 x 64)들이 생성된다. 활성화시키기 위해 ReLU 함수가 적용된다. ReLU함수는 마지막 16층을 제외하고는 항상 적용되니 이 또한 다음 층부터는 설명을 생략하겠다. 

2) 2층(conv1_2): 64개의 3 x 3 x 64 필터커널로 특성맵을 컨볼루션해준다. 결과적으로 64장의 224 x 224 특성맵들(224 x 224 x 64)이 생성된다. 그 다음에 2 x 2 최대 풀링을 stride 2로 적용함으로 특성맵의 사이즈를 112 x 112 x 64로 줄인다. 

*conv1_1, conv1_2와 conv2_1, conv2_2등으로 표현한 이유는 해상도를 줄여주는 최대 풀링 전까지의 층등을 한 모듈로 볼 수 있기 때문이다.  

3) 3층(conv2_1): 128개의 3 x 3 x 64 필터커널로 특성맵을 컨볼루션해준다. 결과적으로 128장의 112 x 112 특성맵들(112 x 112 x 128)이 산출된다. 

4) 4층(conv2_2): 128개의 3 x 3 x 128 필터커널로 특성맵을 컨볼루션해준다. 결과적으로 128장의 112 x 112 특성맵들(112 x 112 x 128)이 산출된다. 그 다음에 2 x 2 최대 풀링을 stride 2로 적용해준다. 특성맵의 사이즈가 56 x 56 x 128로 줄어들었다.

5) 5층(conv3_1): 256개의 3 x 3 x 128 필터커널로 특성맵을 컨볼루션한다. 결과적으로 256장의 56 x 56 특성맵들(56 x 56 x 256)이 생성된다.  

6) 6층(conv3_2): 256개의 3 x 3 x 256 필터커널로 특성맵을 컨볼루션한다. 결과적으로 256장의 56 x 56 특성맵들(56 x 56 x 256)이 생성된다. 

7) 7층(conv3_3): 256개의 3 x 3 x 256 필터커널로 특성맵을 컨볼루션한다. 결과적으로 256장의 56 x 56 특성맵들(56 x 56 x 256)이 생성된다. 그 다음에 2 x 2 최대 풀링을 stride 2로 적용한다. 특성맵의 사이즈가 28 x 28 x 256으로 줄어들었다. 

8) 8층(conv4_1): 512개의 3 x 3 x 256 필터커널로 특성맵을 컨볼루션한다. 결과적으로 512장의 28 x 28 특성맵들(28 x 28 x 512)이 생성된다. 

9) 9층(conv4_2): 512개의 3 x 3 x 512 필터커널로 특성맵을 컨볼루션한다. 결과적으로 512장의 28 x 28 특성맵들(28 x 28 x 512)이 생성된다. 

10) 10층(conv4_3): 512개의 3 x 3 x 512 필터커널로 특성맵을 컨볼루션한다. 결과적으로 512장의 28 x 28 특성맵들(28 x 28 x 512)이 생성된다. 그 다음에 2 x 2 최대 풀링을 stride 2로 적용한다. 특성맵의 사이즈가 14 x 14 x 512로 줄어든다.

11) 11층(conv5_1): 512개의 3 x 3 x 512 필터커널로 특성맵을 컨볼루션한다. 결과적으로 512장의 14 x 14 특성맵들(14 x 14 x 512)이 생성된다. 

12) 12층(conv5_2): 512개의 3 x 3 x 512 필터커널로 특성맵을 컨볼루션한다. 결과적으로 512장의 14 x 14 특성맵들(14 x 14 x 512)이 생성된다.

13) 13층(conv5-3): 512개의 3 x 3 x 512 필터커널로 특성맵을 컨볼루션한다. 결과적으로 512장의 14 x 14 특성맵들(14 x 14 x 512)이 생성된다. 그 다음에 2 x 2 최대 풀링을 stride 2로 적용한다. 특성맵의 사이즈가 7 x 7 x 512로 줄어든다.

14) 14층(fc1): 7 x 7 x 512의 특성맵을 flatten 해준다. flatten이라는 것은 전 층의 출력을 받아서 단순히 1차원의 벡터로 펼쳐주는 것을 의미한다. 결과적으로 7 x 7 x 512 = 25088개의 뉴런이 되고, fc1층의 4096개의 뉴런과 fully connected 된다. 훈련시 dropout이 적용된다.

15) 15층(fc2): 4096개의 뉴런으로 구성해준다. fc1층의 4096개의 뉴런과 fully connected 된다. 훈련시 dropout이 적용된다. 

16) 16층(fc3): 1000개의 뉴런으로 구성된다. fc2층의 4096개의 뉴런과 fully connected된다. 출력값들은 softmax 함수로 활성화된다. 1000개의 뉴런으로 구성되었다는 것은 1000개의 클래스로 분류하는 목적으로 만들어진 네트워크란 뜻이다. 



###          c. ResNet - 신준하
ResNet은 VGGnet-19 구조에서 더 나은 성능을 위해 층의 깊이만을 증가시켰지만, 합성곱 층이 20층 이상으로 깊어질수록 Vanishing/Exploding gradient 현상이 발생하여 이를 해결하기 위해 Shortcut connection(Skip connection)을 적용해 성능을 향상시킨 신경망이다.

![ResNet Block](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_Block.PNG?raw=true)

위 그림은 ResNet에 새롭게 추가 된 skip connection 구조이다. 두 개의 합성곱 층을 지나는 출력값과 그 층을 skip한 원래 값을 더한 후 Activation function을 지나 출력되게 된다.
기본적으로 F(x) + x의 수식을 사용하게 되고, F(x)와 x의 차원이 달라질 경우 논문에서는 zero padding을 진행하거나 F(x) + Wx라는 수식을 사용하며, 위 그림은 stride = 2인 1 x 1 convolution을 진행하는 모습이다.

![ResNet-34](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet-34.PNG?raw=true)

![ResNET-34 Architecture](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet-34_Architecture.PNG?raw=true)

전체적인 ResNet의 구조는 아래와 같다.

0) Input layer : 기존 VGGnet 구조를 따르기 때문에 Input크기는 224 x 224 x 3 이미지(224 x 224 RGB 이미지)를 입력받을 수 있다.

1) 1층(conv1) : 이미지를 64개의 7 x 7 커널필터를 이용하여 컨볼루션한다. zero padding을 3만큼 진행하고, stride = 2로 인해 특성맵의 크기는 224 x 224의 절반인 112 x 112로 출력된다. 여기서 모든 합성곱층의 활성화 함수는 ReLU를 이용한다.

2) 1층 -> 2층(maxpooling) : 입력받은 112 x 112 x 64 특성맵을 3 x 3커널을 이용해 max pooling한다. stride = 2로 인해 특성맵의 크기는 56 x 56으로 출력된다.

3) 2층 ~ 7층(conv2_1 ~ conv2_6) : 입력받은 56 x 56 x 64 특성맵을 3 x 3 커널 64개로 컨볼루션한다.(stride = 1, samepadding 진행) 2개의 합성곱 층 묶음마다 ResNet의 핵심인 skip connection이 진행된다.

4) 8층 ~ 9층(conv3_1 ~ conv3_2) : 입력받은 56 x 56 x 64 특성맵을 3 x 3 커널 128개로 컨볼루션한다. 8층에서는 stride = 2로 인해 특성맵의 크기는 28 x 28로 출력된다. 8 ~ 9층에서의 skip connection이 진행될 때, 입력차원(56 x 56)과 F(x)(28 x 28)의 크기를 맞춰주기 위해 56 x 56 x 64 입력 특성맵을 1 x 1 conv, stride = 2를 이용해 특성맵의 크기와 차원(채널)을 맞춰준다.

5) 10층 ~ 15층(conv3_3 ~ conv3_8) : 28 x 28 x 128 특성맵을 3 x 3 커널 128개로 컨볼루션한다.(stride = 1, samepadding 진행) 2개의 합성곱 층 묶음마다 skip connection이 진행된다.

6) 16층 ~ 17층(conv4_1 ~ conv4_2) : 입력받은 28 x 28 x 128 특성맵을 3 x 3 커널 256개로 컨볼루션한다. 16층에서는 stride = 2로 인해 특성맵의 크기는 14 x 14로 출력된다. 16 ~ 17층에서의 skip connection이 진행될 때, 입력차원(28 x 28)과 F(x)(14 x 14)의 크기를 맞춰주기 위해 28 x 28 x 128 입력 특성맵을 1 x 1 conv, stride = 2를 이용해 특성맵의 크기와 차원(채널)을 맞춰준다.

7) 18층 ~ 27층(conv4_3 ~ conv4_12) : 14 x 14 x 256 특성맵을 3 x 3 커널 256개로 컨볼루션한다.(stride = 1, samepadding 진행) 2개의 합성곱 층 묶음마다 skip connection이 진행된다.

8) 28층 ~ 29층(conv5_1 ~ conv5_2) : 입력받은 14 x 14 x 256 특성맵을 3 x 3 커널 512개로 컨볼루션한다. 28층에서는 stride = 2로 인해 특성맵의 크기는 7 x 7로 출력된다. 28 ~ 29층에서의 skip connection이 진행될 때, 입력차원(14 x 14)과 F(x)(7 x 7)의 크기를 맞춰주기 위해 14 x 14 x 256 입력 특성맵을 1 x 1 conv, stride = 2를 이용해 특성맵의 크기와 차원(채널)을 맞춰준다.

9) 30층 ~ 33층(conv5_3 ~ conv5_6) : 7 x 7 x 512 특성맵을 3 x 3 커널 512개로 컨볼루션한다.(stride = 1, samepadding 진행) 2개의 합성곱 층 묶음마다 skip connection이 진행된다.

10) 33층 -> 34층(global average pooling) : 입력받은 7 x 7 x 512 특성맵을 global average pooling 하여 512개의 채널을 1차원으로 flatten 시켜준다.

11) 34층(FC1) : 1 x 512이 1000개의 dense에 Fully connected 되어 softmax 함수를 적용한다. 이는 1000개의 클래스로 분류를 하겠다는 의미이다.

###          d. Ours - 곽민창
Dataset을 팀원끼리 직접 촬영하고, Processing 하여 사용하기 때문에, Sequential 모델을 사용해 여러 층의 모델을 직접 설계하여 몇 가지 case로 실험을 진행하였다. 각 모델들의 구조를 변화하며 생기는 Accuracy 및 Loss의 변화를 관찰하며 최대한 우리의 Dataset에 맞춘 가볍고 성능이 좋은 Model을 설계하려 노력했다.

처음은 Depth의 변화를 위주로 모델의 성능차이를 비교하였다.
filter_size는 5x5로 고정시켜 놓고, 커널의 갯수는 입력시 32개
이후로는 64개를 고정 시키며 Layer를 쌓았다.
기존의 ReLU보다 성능이 좋은 (Vanishing gradient를 피하기 위함)
ELU함수를 사용하여 skip connection을 대신하였다.

마지막 Dense Layer 이전에는 Global Average Pooling을 통해
너무 많은 파라미터는 제한하였다.

![case1-4](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/case1-4.png?raw=true)

모델을 시각화 하기 위해 Netron을 사용했다.


Case 1) 2 Layer

2개의 Conv2D Layer를 설계하였다. Convolution layer 이후에는 각각 max pooling 한다.
각각의 Activation Function은 ELU를 사용했으며, 출력 전 Dense 하는 것으로 구성되어있다.

Case 2) 3 Layer

3개의 Conv2D Layer를 설계하였다. Convolution layer 이후에는 각각 max pooling 한다.
각각의 Activation Function은 ELU를 사용했으며, 출력 전 Dense 하는 것으로 구성되어있다.

Case 3) 4 Layer

4개의 Conv2D Layer를 설계하였다. Convolution layer 이후에는 각각 max pooling 한다.
각각의 Activation Function은 ELU를 사용했으며, 출력 전 Dense 하는 것으로 구성되어있다.

Case 4) 5 Layer

5개의 Conv2D Layer를 설계하였다. Convolution layer 이후에는 각각 max pooling 한다.
각각의 Activation Function은 ELU를 사용했으며, 출력 전 Dense 하는 것으로 구성되어있다.

![case5~6](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/case5~6.PNG?raw=true)

Case 5) Activation Function : Relu

Case 3에서 사용했던 3 Layer의 Sequential 구조의 각 Convolution layer의 Activation Function을 Relu 함수로 변경하여 구성하였다. 
Activation Function 이외의 모든 변인은 통제되어있기 때문에 Activation Function에 따른 Accuracy와 Loss 값의 차이를 알 수 있다.

Case 6) Activation Function : Sigmoid

Case 3에서 사용했던 3 Layer의 Sequential 구조의 각 Convolution layer의 Activation Function을 Sigmoid 함수로 변경하여 구성하였다. 
Activation Function 이외의 모든 변인은 통제되어있기 때문에 Activation Function에 따른 Accuracy와 Loss 값의 차이를 알 수 있다.


#  4. Evaluation & Analysis
##        - Evaluation : 각각의 모델의 validation accuracy 및 실전 test accuracy 기술
###                              a. AlexNet - 홍노준

![Alex1](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/alex1.png)

 AlexNet의 테스트를 위해 Google Colab에서 Tensorflow framework를 활용한다.

![Alex2](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/alex2.png)

 label별로 순서대로 data를 불러오고, 이후 X에는 img(이미지), Y에는 label(정답값)을 리스트형태로 저장한 후, class는 cafe는 1부터 10, restaurant는 11부터 20이 되도록 설정했다. 원본 DataSet의 이미지는 1440 * 1440 * 3의 형태이기 때문에 OpenCV를 이용하여 224 * 224 * 3으로 변환했다.

![Alex3](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/alex3.png)

 분류를 위해 원핫인코딩을 했다. 전의 데이터셋을 불러오면서 인덱스를 1씩 더하고 시작했기 때문에 인덱스를 맞추기 위해, 각각의 인덱스를 –1씩 뺐다.

![Alex4](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/alex4.png)

![Alex5](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/alex5.png)

전체적인 모델의 구조도를 model.summary()를 통해 확인했다. 전체 파라미터는 약 7.1억 개이며, 마지막 레이어는 총 20개의 클래스를 softmax함수를 통해 분류했다.

![Alex7](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/alex7.png)

10번 훈련한 model의 훈련 성능입니다. validation accuracy가 94.64%까지 나오는 모습을 확인할 수 있다.



###                              b. VGG-19 - 배성현

코드 및 실험 결과에 따른 분석으로 진행하였다.

![vgg_model1](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/vgg_model1.PNG?raw=true)

실험은 Google Colab을 이용한 Tensorflow framework 내에서 진행하였고 , 기존에 있던 MNIST관련 코드를 변형했기에
불필요한 라이브러리를 import하는 코드가 있지만 , 크게 영향이 없어 그대로 진행하였다.

![vgg_model2](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/vgg_model2.PNG?raw=true)

처음으로 Google drive 내의 실험을 위한 Custom dataset을 불러오는 작업을 진행하였다.
Opencv를 이용하여 이미지를 불러온 뒤, 연산량을 줄이기 위해 이미지의 크기를 224로 줄였다.
(추가적으로 VGGNet은 Input Size가 224로 되어있기에 , 이 또한 해당사항이다. 원본 이미지는 1440x1440)
그리고 클래스는 총 20종이기 때문에 num_classes 자체도 20으로 지정해주었다.

![vgg_model3](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/vgg_model3.PNG?raw=true)

불러와진 이미지를 확인해보니 Opencv로 인해 RGB가 아닌 BGR 로 불러와진 것을 확인하였다.
이는 나중에 test때도 동일할 것으로 판단되어 크게 신경쓰지 않기로 하였다.
불러온 dataset을 0.64 : 0.16 : 0.2 로 train_val_test split을 진행하였다.
label은 총 20개로 분류모델을 위해 원핫인코딩을 진행해주었다.
위의 데이터셋을 불러오는 부분에서 인덱스를 1부터 시작했지만 , 원핫인코딩은 0번인덱스부터 시작하기에
각각의 인덱스를 -1씩 빼준 모습이다.

![vgg_model4](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/vgg_model4.PNG?raw=true)

위의 VGG-16 분석과 동일하게 Depth,kernel_size,activation,FCL 등을 설정한 모습이다.
논문내의 모델과 완전 동일하다고 설명할 순 없지만 , 최대한 유사하게 구현한 모습이다.

![vgg_model5](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/vgg_model5.PNG?raw=true)

전체적인 모델의 구조도를 model.summary()를 통해 확인한다.
전체 파라미터는 약 1.3억개이고 , 마지막 레이어는 총 20개의 클래스를 softmax함수를 통해 분류한다.

![vgg_model6](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/vgg_model6.PNG?raw=true)

학습 결과를 보면 , AlexNet보다 낮은 accuracy와 업데이트가 되지않는 모습을 보인다.
이는 적은 dataset을 너무 깊은 depth를 통해 구현하였기에 생긴 vanishing gradient 문제라고 판단된다.
이를 통해 단순한 Task를 위한 모델은 dept를 너무 깊게하지 않으며 , 혹은 이후에 다루게 될 ResNet과 같은
Technique을 사용해야 한다는 것을 알았다.

###                              c. ResNet - 신준하
코드는 google의 colab으로 진행하였고, 주요 코드마다 결과를 확인하였다.

![image](https://user-images.githubusercontent.com/87685924/205059104-9f8d6d4a-81e5-4470-8a19-230c6b0b4759.png)

필요한 라이브러리들을 import 해주었고, google drive에 저장해 놓은 dataset을 이용하기 위해 drive 모듈로 drive mount 해주었다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_2.PNG?raw=true)

훈련에 이용 될 dataset 위치

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_3.PNG?raw=true)

label별로 순서대로 data를 불러오고, 이미지 크기를 224 x 224 3channel로 맞춰주었다. 이후 X에는 img(이미지), Y에는 label(정답값)을 리스트형태로 저장한 후 np.array를 이용해 배열로 바꿔주었다.
class는 cafe는 1부터 10, restaurant는 11부터 20이 되도록 설정해주었다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_4.PNG?raw=true)

모델의 input의 형태는 224 x 224 x 3, batchsize = 4, class개수는 20(cafe 10 + restaurant 10)으로 맞춰주었다.
이후 X내에 있는 data img들 중 22번 index의 이미지를 확인해보았다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_5.PNG?raw=true)

X와 Y에 있는 데이터를 split함수를 이용해 train : validation : test 비율을 64:16:20으로 나누었다.
이후 각 데이터셋의 값들을 255로 나누어 주고, 분류를 위해 one-hot encoding을 실시했다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_6.PNG?raw=true)

model구성은 keras에 있는 ResNet50을 불러와 사용했고, 마지막 pooling층은 GlobalAveragePooling으로 설정하였다. 이때 FullyConected층이 없어 3개의 FC층을 추가로 만들어 주었고 상위 두 FC층은 relu, 마지막 FC층은 softmax를 이용하였다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_7.PNG?raw=true)

                                                  .

                                                  .

                                                  .

![image](https://raw.githubusercontent.com/hunction/AI-X-DeepLearning/main/Markdown_Img/ResNet_8.PNG)

model의 summary이다. 앞서 말한대로 ResNet50, GlobalAverage, FC3층이 포함되어 있는 것을 확인 할 수 있다. 파라미터는 총 25,958,548개 이며, 학습되는 파라미터 수는 25,902,868개 이다.

![image](https://raw.githubusercontent.com/hunction/AI-X-DeepLearning/main/Markdown_Img/ResNet_9.PNG)

model의 훈련을 위해 사용 할 loss는 분류를 위한 CE이고, optimizer는 adam, 성능 확인 기준은 정확도로 설정하였다. epoch은 150회로 설정하였고, batch_size는 위에서 설정한대로 사용한다. 이때 한 epoch 훈련 후 모델에 validation dataset을 넣어 test 성능을 파악하였다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_10.PNG?raw=true)

150회의 epoch가 끝나기 직전 11번의 model 훈련 성능이다. validation accuracy가 100%까지 나오는 모습을 확인 할 수 있다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_11.PNG?raw=true)

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_12.PNG?raw=true)

성능을 확인하기 위해 정확도와, loss를 시각적으로 표현하였다. 전체적으로 train은 튀는 현상 없이 꾸준하게 정확도는 증가하고 loss는 감소하면서 두 값 모두 수렴하는 것을 볼 수 있다.
하지만 validation에 대한 정확도와, loss는 꾸준히 좋은쪽으로 수렴하고 있지만, 가끔씩 튀는 현상이 일어났다.
train상태에서 어느정도 epoch 이후로는 거의 수렴하여 변화가 없는것으로 보였는데, 이는 epoch를 조금 더 줄여서 훈련을 시켜도 되는 것으로 판단이 되었다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/ResNet_13.PNG?raw=true)

훈련시킨 model을 test 데이터를 이용하여 평가해보았다. 성능이 좋게 나왔고, 이 모델의 아키텍쳐와 가중치들을 가지고있는 파일을 저장하였다.

###                              d. Ours - 곽민창
코드는 Jupyter notebook을 사용하였고, 각 Case에 맞는 코드를 작성하여 각각 학습을 진행하고, matplotlib을 이용하여 History를 시각화하였다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Ours_1.png?raw=true)

필요한 라이브러리들을 import 해 주었다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Ours_2.png?raw=true)

이미지를 불러온 뒤, 해당 폴더명과 같은 라벨 Y를 반환해 주었다. 또한, 원본 DataSet의 이미지는 1440 * 1440 * 3의 형태이기 때문에 OpenCV를 이용하여 200 * 200 * 3으로 변환해주었다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Ours_3.png?raw=true)

DataSet에서 나뉘어있던 식당과 카페의 이미지 데이터를 합치고, 각각 Label을 매겨 학습할 수 있도록 Numpy Array로 만들어 주었다. 그후 원핫 인코딩을 진행하였다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Ours_4.png?raw=true)

학습에 대한 검증은 validation data로, 실전 정확도는 test data로 검증하기 위하여 train_test_split을 하여 데이터를 train, validation, test로 나누어주었다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Ours_5.png?raw=true)

원본 DataSet의 개수가 부족하기 때문에 증강 기법을 적용하였다. ImageGenerator를 이용하여 Data를 Augumentation 해주고, 매 Epoch 마다 다른 이미지 데이터가 들어가도록 했다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Ours_6.png?raw=true)

Sequential Model에 각 앞서 나눴던 Case에 맞는 모델을 설계해 구성하였다. 각 convolution layer에는 elu activation function을 사용하였다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Ours_7.png?raw=true)

첫 번째 Case에서의 모델 구성 Summary이다. 총 Parameter는 54,996개를 사용하는 것을 볼 수 있다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/Ours_8.png?raw=true)

Adam Optimizer를 사용하였고, 학습은 batch : 32, epoch : 10으로 진행하였다. 학습 History는 모두 hist Array에 저장하도록 하였다.

[case 1] 2 - Layer

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/case1.png?raw=true)

[case 2] 3 - Layer

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/case2.png?raw=true)

[case 3] 4 - Layer

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/case3.png?raw=true)

[case 4] 5 - Layer

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/case4.png?raw=true)

[case 5] Activation Function : ReLU

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/case5.png?raw=true)

[case 6] Activation Function : Sigmoid

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/case6.png?raw=true)

[결과 정리]

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/result_1.png?raw=true)
![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/result_2.png?raw=true)

Convolution Layer의 개수를 변화하여 학습을 진행했던 Case 1 ~ 4의 Epoch별 Accuracy와 Loss를 나타낸 그래프는 다음과 같다. Convolution Layer가 많아질수록 더 빨리 Accuracy가 오르고, Loss는 줄어드는 것을 확인할 수 있다. 또한, 10 Epoch 후의 결과 또한 Convolution Layer가 5개 일때 가장 이상적인 것을 볼 수 있다.

![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/result_3.png?raw=true)
![image](https://github.com/hunction/AI-X-DeepLearning/blob/main/Markdown_Img/result_4.png?raw=true)

Activation Function을 ELU, ReLU, Sigmoid로 변화하여 학습을 진행했던 Case 2, 5, 6의 Epoch 별 Accuracy와 Loss를 나타낸 그래프는 다음과 같다. ELU와 ReLU Activation Function을 사용했을 때 Accuracy와 Loss 모두 비슷하게 좋은 결과를 보여주고 있지만, Sigmoid Activation Function을 사용했을 경우 결과가 좋지 않은 것을 볼 수 있다.
        
##        - Analysis : 모델 별 학습 파라미터의 갯수 및 학습시간,  동작시간 기술
        
#  5. Related Works
- 각각의 논문 제목 기술 및 네트워크 생성시 참조한 블로그

AlexNet :        코딩 관련 블로그 - https://bskyvision.com/421, AlexNet 구조 - https://ctkim.tistory.com/120

VGG-19 :            논문분석 블로그 - https://bskyvision.com/504    네트워크 생성 - https://minjoos.tistory.com/6
        
ResNet50 :          네트워크 설명 - https://jisuhan.tistory.com/71     논문 - "Deep Residual Learning for Image Recognition"(arXiv:1512.03385v1 [cs.CV] 10 Dec 2015)

#  6. Conclusion: Discussion
- 하이퍼 파라미터의 변화에 따른 각각의 성능변화에 대한 정리

총 세가지의 기존 모델을 분석하고 실험한 결과 ,
무조건 적인 depth의 증가는 오히려 양날의 검이 될 수 있다는 것을 알게 되었다.

따라서 , 우리는 너무 깊지 않은 네트워크 구조와 이후에 실험한 활성함수에 대한 결과를 적용하기로 하였다.
나름 최근에 등장한 ELU 함수보다 ReLU 함수가 더 좋은 결과를 보여주었다.
Skip connection의 경우 , 층이 너무 깊지 않아 사용하지 않기로 하였다.
( 사용할 경우 오버피팅의 요소가 있을 것이라 판단 )
GlobalAveragePooling은 파라미터수를 많이 감소시켜주어 기존의 목적인 어플리케이션 내 탑재를 가능케 할 것이라 생각했다.

단순한 Task이기 때문에 , 많지 않은 층으로 적은 에포크로 충분한 Saturation을 보여주는 것을 확인했다.

향후에는 이를 적용한 어플리케이션을 만드는 것이 우리팀의 목표이다!

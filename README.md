# AI-X-DeepLearning
### Final Project


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
그림 및 설명

###          b. VGG-19 - 배성현
그림 및 설명

###          c. ResNet - 신준하
그림 및 설명

###          d. Ours - 곽민창
그림 및 설명
          
          
#  4. Evaluation & Analysis
##        - Evaluation : 각각의 모델의 validation accuracy 및 실전 test accuracy 기술
###                              a. AlexNet - 홍노준
그림 및 설명
###                              b. VGG-19 - 배성현
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


###                              c. ResNet - 신준하
그림 및 설명
###                              d. Ours - 곽민창
그림 및 설명
        
##        - Analysis : 모델 별 학습 파라미터의 갯수 및 학습시간,  동작시간 기술
        
#  5. Related Works
- 각각의 논문 제목 기술 및 네트워크 생성시 참조한 블로그
VGG-19 : https://bskyvision.com/504
        
        
#  6. Conclusion: Discussion
- 하이퍼 파라미터의 변화에 따른 각각의 성능변화에 대한 정리

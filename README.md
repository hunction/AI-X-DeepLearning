# AI-X-DeepLearning
Final Project


# Title : Classifier For Freshman 

# Members :
          Bae Sung Hyun , Electronic Engineering , hyung50300@gmail.com
          Shin
          Guak

# Index
          1. Proposal
          2. DataSets
          3. Methodology
          4. Evaluation & Analysis
          5. Related Works
          6. Conclusion: Discussion
          

          
#  1. Proposal ( Option A )
        - Motivation : When Senior comes to Univ at first , He does not know what kind of food and drinks did those restaurants & cafes sell.
                       So , We make this Classifier to explain menus and how much is it.
                       For those things, the first step is 'Do classify' with some pictures to descride a place info.
                                                            
                                                            +
                                                            
                       Make a Simple Deep Neural Network for educating our team member ( Freshman, Electronic Engineering )
                       
        - Start / End ( Pipe line ) : Someone takes a input picture to this software , it will classify label of picture like 'StarBucks'
                                              Picture -> Label ( To describe its info )
           
           
#  2. DataSets
        - Professor said that you could use some kaggle datasets for training but we did not use that.
          Just take some pictures on our hand ( 10 Restaurants & 10 cafes for 15 pictures for each class )
          Each pictures have different resolutions , so we take some preprocessing to downscaling their resolutions ( and make rectangular to square )
          And then take some augmentation technique to make datasets larger. ( 20 * 15  -> 20 * 15 * 10 ) - rotate images or shifting images
          
          So , we used 3000 images for training Network ( each class have 150 images. left , front , right view images ) 
          
#  3. Methodology
        - Just use tensorflow.kears 's Dense / activation / Conv2D Layers for making Networks.
          we have some experiments that change some hyperparameter such as filter size , activation functions , Network depth etcs.
          And then compare those experiments with other state-of-the-arts networks.
          
          
          
          
#  4. Evaluation & Analysis
          
          

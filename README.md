# AI-X-DeepLearning
Final Project

# Title : Classifier For Seneior 

# Members :
          Bae Sung Hyun , Electronic Engineering , hyung50300@gmail.com
          Shin
          Guak
          
          
#  1. Proposal ( Option A )
        - Motivation : When Senior comes to Univ at first , He does not know what kind of food and drinks did those restaurants & cafes sell.
                       So , We make this Classifier to explain menus and how much is it.
                       For those things, the first step is 'Do classify' with some pictures to descride a place info.
                       
        - Start / End ( Pipe line ) : Someone takes a input picture to this software , it will classify label of picture like 'StarBucks'
                                              Picture -> Label ( To describe its info )
           
           
#  2. DataSet 
        - Professor said that you could use some kaggle datasets for training but we did not use that.
          Just take some pictures on our hand ( 10 Restaurants & 10 cafes for 15 pictures for each class )
          And then take some augmentation technique to make datasets larger. ( 20 * 15  -> 20 * 15 * 10 ) - rotate images or shifting images
          
          So , we used 3000 images for training Network
          
#  3. Methology
        - Just use tensorflow.kears 's Dense / activation / Conv2D Layers for making Networks.
          we have some experiments that change some hyperparameter such as filter size , activation functions , Network depth etcs.
          
          
          

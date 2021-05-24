# siamese_conv_net

Siamese network refers to the use of two networks having the same structure as a pair.
Each pair has the same weight, and each feature vector is calculated from the two input factors.
Assume that the input images are image1 and image2, respectively.
If the two images are of the same class, that is, similar images, the feature vectors from each neural network(already trained) will have similar values, 
and the distance between these vectors will also be close.
Conversely, if the two images are of different classes, the similarity between the feature vectors will be low and the distance will have a large value.
Using these features, we can set a kind of boundary value between two feature vectors,
and classify images into different classes if the distance is greater than or equal to the boundary value, 
and the same class if the distance is less than the boundary value.
In addition, this project uses thermal facial images of humans as a dataset, and some of these datasets are provided as sample data.

training  
-----------------------
As mentioned earlier, the trained neural network should have a 
weight value that makes the distance between feature vectors in different classes far and close in the same class.
The training method of this network also defines a constrative loss function that trains two images to be closer 
if they are of the same class and farther if they are of the same class.
Divide the provided sample dataset into training data and test data (you can use data_mover.py), and proceed with training using only the training data.
You can train the model by using the siamese_train.py file like the command below, and you need to specify some variable values before executing the code.

    python siamese_train.py


The number of classes, the number of epochs, the path of the training image, and the shape of the input must be specified in the code as shown below.
If you use the sample dataset as it is, the image size is already (60, 80, 3), so there is no need to touch this part separately. 
If you want to resize the image and use it, you need to change the part to properly generate the training model.

    num_classes = 20
    epochs = 40
    datagen = data_generator(DATASET_PATH = 'C:/projects/dataset/thermal_image/80_60_origin_gray_image/', shuffle_sel=True)
    input_shape = (60, 80, 3)
    
As the learning progresses, you can check the training progress epoch and training & validation accuracy as shown in the figure in the command window.  
  
![train](https://user-images.githubusercontent.com/84235639/119297708-76d94980-bc96-11eb-92e4-19286c51bae2.JPG)

tsne_plot
--------------------
    python plot_tsne_siamese.py  
    
TSNE is just a tool for visualization, but since there is a part that selects representative data through this code, execution must be preceded before the test code.
In this implemented code, when a specific image is input, each representative image in the class is compared with the input image once, 
and the closest class is determined as the class of the input image.
When determining the representative image,
the image at the center of the class is selected as the representative image by calculating the distance between the two-dimensional coordinates reduced by TSNE. 
Since this process is included in the process of plotting with TSNE, this code must be executed first.


    flags.DEFINE_string('MODEL_DIR', 'model.40-0.01.h5', 'Directory to put the model.(.h5)')
    flags.DEFINE_string('TEST_DATA_DIR', 'C:/projects/dataset/thermal_image/80_60_test_dataset/', 
                    'Directory to put the test data.')
    flags.DEFINE_string('REP_DATA_DIR', 'C:/projects/dataset/thermal_image/80_60_rep_dataset/', 
                    'Directory to put the representative data.')  


There are some parts to be specified in the TSNE code as well. 
The path of the train data and the path to store representative data must be specified, and the path of the model that has been trained must also be specified.
If you run the code after specifying the path, you can see a picture of the high-dimensional feature vector projected onto a two-dimensional plane as shown below. 
Data displayed in dark black indicate the location of representative data for each class.
The figure below is a picture tested with 20 classes, and the dataset provided as a sample is a smaller amount of classes.

![train_tsne](https://user-images.githubusercontent.com/84235639/119297711-76d94980-bc96-11eb-810c-cbe9d6b900a7.jpeg)  

test
------------------
    python siamese_test.py  
    
Once the representative data have been selected from the training data, you can now test the model with data that has never been used for training.  

    flags.DEFINE_float('DISTANCE_THRESHOLD', 0.2, 'class judgement threshold distance.')
    flags.DEFINE_string('TEST_DATA_DIR', 'C:/projects/dataset/thermal_image/80_60_test_dataset/',
                'Directory to put the test data.')
    flags.DEFINE_string('REP_DATA_DIR', 'C:/projects/dataset/thermal_image/80_60_rep_dataset/',
                'Directory to put the representative data.')
    flags.DEFINE_string('MODEL_DIR', 'model.40-0.01.h5',
                'Directory to put the model.(.h5)')
    flags.DEFINE_boolean('TEST_ANOMALY', False, 'test anomaly classes = True, know classes = False.')
    

In the same way, several configuration paths and values are required. 
DISTANCE_THRESHOLD is a threshold value determined by the same class, and the default value is set to 0.2.
In addition, the path of the test image and the path of the representative image (the representative image selected from the TSNE code) must be given as input factors.
The TEST ANOMALY option is a factor that determines whether to test for unseen class data.
If you want to see whether the learned class is properly classified as in the current situation, 
select the option with FALSE. For example, 
if you have learned about classes 1 to 10 and want to check whether external classes such as 
11 and 12 are properly filtered (whether feature vectors with all classes have a larger value than the boundary value), set this option to TRUE Just set it up.
As a result of executing the code, it will be expressed how accurate it has for the test data.
And the picture visualized using TSNE for this test data is as follows. 
It can be seen that the separation between classes is well done, and from this, it can be seen that the weight value of the model is well learned.

![test_tsne](https://user-images.githubusercontent.com/84235639/119297704-75a81c80-bc96-11eb-925d-b94ece195384.jpeg)

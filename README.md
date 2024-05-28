# A Study on Extending Homomorphic Encryptions to Machine Learning

Note: Read the Project_Report.pdf for more details.

The code in this repository implements a three layer fully-connected feed-forward neural network. It has an input layer and a hidden layer with 12 nodes each. The output layer has 4 nodes to classify the data into four categories [0, 90, 180, 270]. I used 1-hot encoding scheme to encode the output labels. For each image from the input dataset, the pixel values at indices [2, 47, 185, 119, 122, 170, 146, 173, 176, 149, 191, 188] are the features of interest considered by the neural network during learning and classification. The input to the neural network is fed using the values retrieved from the pixel data using the above list. These indices are the nodes at the top three layers of a decision tree which was obtained by calculating the minimum entropy.

The accuracy of the classification done by neural networks is as follows.

```
Correctly classified: 668 out of 943
Incorrectly classified: 275 out of 943
Classification accuracy: 70.84%
```

To run the code please use the following command.
```
Training:
python3.7 orient.py train train-data.txt nnet_model.txt nnet

Testing:
python3.7 orient.py test test-data.txt nnet_model.txt nnet
```

The execution takes around 13 mins if the network is evaluated homomorphically. Otherwise it takes less than 1 sec (to disable homomorphic evalution please change the line:536 from he_classify to classify in neural_networks.py file). 

I used Python3.7 built using GCC 7.4.0 on Ubuntu-16.04 to run the code.

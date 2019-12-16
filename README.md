# KaggleChallenges
Here I will upload my notebooks for different Kaggle challenges!


# Toxic Comment Classification - Score: 97.464%. (1.4% off from the best solution)
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

In this challenge I have used a multilayered neural network with an GloVe embedding layer, LSTM layer and GRU layer. In addition to the embedding layer, I have implemented two features. One feature represent the amount of exlamation marks used in the sentence and the other feature represent the amount of CAPS used in the sentence. These features are scaled between 0 and 1.

# Quora Question pairs  - Score : Averaging 75%.
https://www.kaggle.com/c/quora-question-pairs

For this challenge I have used 3 different approaches. These approaches will not provide the best solution for this challenge. It was interesting to see how these different approaches would perform on this data set. The reason why I did not attempt the "best solution" is that this solution requires alot of effort in specific knowledge about language processing, which I am not interested in. I am interested in a quick solution that is applicable for different problems, instead of creating really specific features that are only relevant for this specific challenge.

First approach - score 75%.
Every sentence is represented as a vector of 20 (20x1). Each word in this vector is a GloVe vector of 100 dimensions, which actually makes the vector a matrix. The difference between every position in the vector is taken and added to a new matrix of 400. Let's call the starting vectors A and B, a1 and b1 being the first element in the vector. The first element of the new matrix, with a dimension of 400x100, is a1-b1. The second element is a1-b2, third element is a1-b3, etc. This matrix is used as input for a CNN. Why is the score only 75%? I suspect that in this CNN approach, the matrix that I created has an underlying spatial component that does not represent the problem very well. The CNN is probably confused by this spatial component.

Second approach - score xx%.
Every sentence is represented as a vector of 20 (20x1). Each word in this vector is a GloVe vector of 100 dimensions, which actually makes the vector a matrix. This matrix of 20x100 is used as input for a auto-encoder. The encoded vector hopefully does not have an underlying spatial component like in the first approach. This encoded vector is substracted from the other encoded sentence vector, which gives us a difference vector. This difference vector will be used as feature for a random forest. Let's find out the results!

Third approach - score xx%.
Every sentence is represented as a vector of 20 (20x1). Each word in this vector is a GloVe vector of 100 dimensions, which actually makes the vector a matrix. This matrix of 20x100 is used as input for a LSTM network. The LSTM will output a vector that hopefully does not have an underlying spatial component like in the first approach. The output vector will be substracted from the other sentence vector that is processed by the LSTM. This difference vector will be used as a feature for a random forest. Let's find out the results!

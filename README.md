# Traffic-Sign-Classifier

The [German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset on kaggle is a popular multi-class classification dataset on Kaggle. Traffic sign data was collected and is to be classified into 43 different classes. The data visualizations can be found in the `visualization.py` file. A small sample of the dataset is shown below. The dataset is diverse, including images which have low contrast, which even humans may find difficult to classify. Some images are shaky, and some are blurred.

![Dataset Sample Images](https://raw.githubusercontent.com/sbalan7/Traffic-Sign-Classifier/master/Images/sample.png)

The distribution of the classes was then plotted with `matplotlib` to get a feel for the dataset balance. However, the data is highly imbalanced with some classes having around 2000 samples, while others having only 200. That's almost an order of magnitude of difference. To balance the dataset, weighting or oversampling could have been done. In this model, the weights were used. To weight the data, the number of images in every class was calculated. The weight was the reciprocal of this value divided by the maximum number of images in a class.

![Class Distribution](https://raw.githubusercontent.com/sbalan7/Traffic-Sign-Classifier/master/Images/images_per_class.png)

The deep learning model was built with TensorFlow and Keras. The training code is in the file `training.py`. The model alternates between a convolution layer and a pooling layer thrice before flattening and using a normal one layered feed forward network. All layers are ReLU activated. The feed forward connects to the output layer with a softmax activation. The model summary is shown below.

![Model Summary](https://raw.githubusercontent.com/sbalan7/Traffic-Sign-Classifier/master/Images/model_summary.png)

The images in the dataset were all of different sizes. They were preprocessed to a uniform size of <img src="https://render.githubusercontent.com/render/math?math=30 \times 30 \times 3">. The output vectors were one-hot encoded. The loss function used was categorical crossentropy, and the model was compiled with the Adam optimizer. The model was finally trained for 20 epochs. The variation of train and validation accuracy and loss as the epochs pass is plotted below. The trained model was saved in a file.

![Train Chart](https://raw.githubusercontent.com/sbalan7/Traffic-Sign-Classifier/master/Images/train.png)

Finally, the model is used in a simple GUI interface built with `PyQt5`. The image is to be loaded into the file, and the prediction is done automatically.

![GUI interface](https://raw.githubusercontent.com/sbalan7/Traffic-Sign-Classifier/master/Images/GUI.png)

The next version of the project can probably include localisation of the sign in an entire image, with a bounding box highlighting the traffic sign.
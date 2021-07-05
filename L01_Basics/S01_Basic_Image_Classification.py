import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

cloth_class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_cloth_name(index: int) -> str:
    return cloth_class_name[index]


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(cloth_class_name[predicted_label],
                                         100 * np.max(predictions_array),
                                         cloth_class_name[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def main():
    # show your tensorflow version
    print("Tensor flow version : {}".format(tf.__version__))

    # get the fashion_mnist data set (which replaces the classic mnist of hand written digit )
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # the load_data method returns four NumPy arrays, which contains features and labels
    # for the model training.
    # Unlike tabular data set, each feature is a column. For image, each pixel is a feature
    # In this example, the images are 28*28 arrays, and the pixel value ranging from 0 to 255.
    # As we do a classification of cloths. The labels has 10 possible values, which is
    # represented as an integer ranging from 0 to 9.

    # Following table shows the correspondence between value and cloth type
    # 0	T-shirt/top
    # 1	Trouser
    # 2	Pullover
    # 3	Dress
    # 4	Coat
    # 5	Sandal
    # 6	Shirt
    # 7	Sneaker
    # 8	Bag
    # 9	Ankle boot
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Step-1 Explore data set
    # The output (60000, 28, 28) tells you data set has 60000 image of size 28*28 pixels
    print("Train data has shape: {}".format(train_images.shape))
    print("Train data label has length: {}".format(len(train_labels)))
    print("Test data has shape: {}".format(test_images.shape))
    print("Test data label has length: {}".format(len(test_labels)))

    # Step-2 Preprocess the data
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print("image 0 is : {}".format(get_cloth_name(train_labels[0])))

    # most of image prediction models only take feature pixel ranging [0,1] or [-1,1]. But image pixel
    # are often ranging [0,255]. So we need to rescale the image pixel
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # To view the result of rescaling
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(cloth_class_name[train_labels[i]])
    plt.show()

    # Step-3 Build the model
    # The basic building block of a neural network is the layer. Layers extract representations from the
    # the data fed into them.

    # 1. we define that our model is a sequence of layers
    model = tf.keras.Sequential()

    # 2. We add first layer which is the input layer, as we know the input layer neurons number
    #     must equal to the input feature number. In our case, the input feature number is the
    #     number of pixels in an image, which is 28*28=784.
    #     In another word, input layer has no parameter to learn, it only reformat the input data
    #     that convert a two-dimensional array(28*28) to a one-dimensional array(784)
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # 3. Second layer is a Dense layer(aka. fully connected layer), which means for each neurons
    #    in this layer, it receives signals from all neurons of the previous layer. We set the
    #    number of neurons as 128 (arbitrary, you can change it to any number that suits you),
    #    and the activation function as relu
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # 4. The output layer is also a Dense layer, which means all neurons in this layer connect
    #    to all 128 neurons of previous layer. We set the neurons number as 10, because we have
    #    10 possible class. As it's the output layer, we do not need an activation function.
    model.add(tf.keras.layers.Dense(10))

    # Step-4: Compile the
    # Loss function: This measures how accurate the model is during training. You want to minimize
    #                this function to "steer" the model in the right direction.
    # Optimizer: This is how the model is updated based on the data it sees and its loss function.
    # Metrics: Used to monitor the training and testing steps.
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Step-5: Train the model
    # We train the model by telling it which image needs to associate to which label.
    # epochs = 10 means we pas 20 times for all images in the training data set to the
    # model. one epoch means "one forward pass and one backward" pass of all the
    # training examples
    model.fit(train_images, train_labels, epochs=10)

    # Step-6: Evaluate model accuracy
    # You can notice the model accuracy on the test dataset is a little less than the accuracy
    # on the training dataset. This gap represents over-fitting of our model on the training
    # datasets.
    # Over-fitting happens when a machine learning model performs worse on new, previously unseen
    # inputs than it does on the training data. An over-fitted model "memorizes" the noise and
    # details in the training dataset to a point where it negatively impacts the performance of
    # For more details on over-fitting:
    # plz visit https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#demonstrate_overfitting
    # For more details on how to prevent over-fitting:
    # plz visit https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy: {}".format(test_acc))

    # Step-7: Apply the model to make a prediction
    # You can notice the raw output has big negative and positive values. We need the prediction
    # result in range of [0,1], which represent the probability of the image belongs to each class
    # To do so, we add a new layer
    raw_prediction = model.predict(test_images)
    print("raw prediction: {}".format(raw_prediction[0]))

    improved_model = tf.keras.Sequential()

    # add the base model that we just trained
    improved_model.add(model)

    # add a softmax layer on top of the output
    improved_model.add(tf.keras.layers.Softmax())
    improved_prediction = improved_model.predict(test_images)
    print("improved prediction: {}".format(improved_prediction[0]))

    # now for each prediction, we have 10 values ranging [0,1] which represent the probability of
    # the image is in the corresponding class.
    # to make more clear, we use np.argmax to return the index of the max value of the 10(most likely
    # class of the image). Then we find the string class corresponding of the index.

    prediction_in_str = cloth_class_name[np.argmax(improved_prediction[0])]
    print("improved prediction in string: {}".format(prediction_in_str))
    print("real label value: {}".format(cloth_class_name[test_labels[0]]))

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, improved_prediction[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, improved_prediction[i], test_labels)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

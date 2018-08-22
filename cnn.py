# Part 1 - Building the CNN

# Importing the Keras libraries and packages
# We use sequential since our nerual network is a sequence of layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from pathlib import Path

# Prepare and seperate the data
home_path = os.getcwd()
training_path = home_path + "/dataset/training_set"
test_path = home_path + "/dataset/test_set"

"""
for root, dirs, files in os.walk(training_path):
   for name in dirs:
       
       new_test_directory = test_path + "/" + name
       if not os.path.exists(new_test_directory):
           os.makedirs(new_test_directory)
          
       num_test_files = 250
       curr_num = 1
       
       for filename in os.listdir(os.path.join(root, name)):
           if curr_num > num_test_files:
               break;
           
           image_name = name+str(curr_num)+".jpg"
           shutil.move(training_path + "/" + name + "/" + image_name, new_test_directory)
           curr_num = curr_num + 1
"""      

# Initializing the CNN
classifier = Sequential();

# Part 2 - Fitting the CNN to the images    
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
test_datagen = ImageDataGenerator(rescale=1./255)
    
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical')
    
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical')

# If we have a saved version of the model, load it
saved_model_file = Path(home_path + "/my_model.h5")

if saved_model_file.exists():
    classifier = load_model('my_model.h5')
else:
    # Step 1 - Convolution
    # We want 32 feature detectors of 3x3.  
    # We will choose 3 channels sicne we are workign with color
    # We'd use 1 if it was black and white
    # As for our input image sizes, we should make them 64x64
    # If we are working on a GPU, we could do something more intense like 256x256
    # In order to get some non-linearity, we need an activation function
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'));
    
    # Step 2 - Pooling
    # This step reduces the size of your feauture maps by half
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    # Add a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'));
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    # Step 3 - Flattening
    # Gets the ouput of the CNN and converts it into a 1D vector to be used
    # as an input in the ANN
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    # We kind of experimented by choosing 128.  Common practice to choose a 
    # number that is a power of 2.
    # We use sigmoid as the activation funciton in the second layer since it is 
    # supposed to have a binary outcome.  If we wanted something that was more than
    # just two categories, then we would need a softmax function
    classifier.add(Dense(activation = 'relu', units = 64));
    classifier.add(Dense(activation = 'softmax', units = 29));
    
    
    # Compiling the CNN
    # What goes inside the optimizer is the stochastic gradient descent algorithm
    # For the loss function, we use binary cross entropy.  If we had more than two
    # categories, we'd use categorial cross entropy.
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']);
    
    
    #Since we have 8000 images in our training set, we need steps_per_epoch to be 8000
    #We set epochs to 25 so that we don't have to wait too long
    #Since we have 2000 images in our test set, we set validation_steps to 2000
    classifier.fit_generator(
            training_set,
            steps_per_epoch=79750,
            epochs=2,
            validation_data=test_set,
            validation_steps=7250)


#Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/asl_alphabet_test/Y_test.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)

#Our classifier can't just take one image, it has to take a batch
#This is the case even if that batch contains one or two images
#We can do this by just adding another dimentions to test_image
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image);
training_set.class_indices

print("HERE IS THE RESULT OF THE SINGLE TEST")
for idx, val in enumerate(result[0]):
    if val == 1:
        for key in training_set.class_indices:
            if training_set.class_indices[key] == idx:
                print(key)



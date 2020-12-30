'''
Knowledge transfer in Neural Networks (VGG16, ResNet50)
'''

import cv2
import numpy as np
from keras.applications import VGG16,ResNet50
#import matplotlib.pyplot as plt
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
# select which image you want to predict
case_selector=1
# load the image from disk and display the width, height and depth
if case_selector==1:
        print('----------- Jake 1 predictions -----------')
        runfile('display_image.py', args='--image "Jake\jake1.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')
elif case_selector==2:
        print('----------- Jake 2 predictions -----------')
        runfile('display_image.py', args='--image "Jake\jake2.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')
elif case_selector==3:
        print('----------- Jake 3 predictions -----------')
        runfile('display_image.py', args='--image "Jake\jake3.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')
elif case_selector==4:
        print('----------- Jake 4 predictions -----------')
        runfile('display_image.py', args='--image "Jake\jake4.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')
elif case_selector==5:
        print('----------- Ermis 1 predictions -----------')
        runfile('display_image.py', args='--image "Ermis\Ermis1.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')
elif case_selector==6:
        print('----------- Ermis 2 predictions -----------')
        runfile('display_image.py', args='--image "Ermis\Ermis2.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')
elif case_selector==7:
        print('----------- Ermis 3 predictions -----------')
        runfile('display_image.py', args='--image "Ermis\Ermis3.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')
elif case_selector==8:
        print('----------- Fred 1 predictions -----------')
        runfile('display_image.py', args='--image "Fred\Fred1.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')
elif case_selector==9:
        print('----------- Fred 2 predictions -----------')
        runfile('display_image.py', args='--image "Fred\Fred2.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')
elif case_selector==10:
        print('----------- Fred 3 predictions -----------')
        runfile('display_image.py', args='--image "Fred\Fred3.jpg" ')
        # resize the data
        image_resized=cv2.resize(image,(224,224))
        # expand the dimensions to be (1, 3, 224, 224)
        image_expanded = image_utils.img_to_array(image_resized)
        image_expanded = np.expand_dims(image_expanded, axis=0)
        image_expanded = preprocess_input(image_expanded)
        
        model = ResNet50(weights="imagenet")
        pred_image = model.predict(image_expanded)
        P = decode_predictions(pred_image)
        # loop over the predictions and display the rank-5 predictions + probabilities to our terminal
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        print('------------------------------------------')


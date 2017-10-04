from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os
import pdb

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    # guided backpropagation, as described in Springenberg et al. 2015
    # simply sets grad = 0 at the indices where grad <= 0 or inputs <= 0
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer="block5_conv4"):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3) # max along feature channel axis?
    saliency = K.gradients(K.sum(max_output), input_img)[0] # K.gradients is an auto-differentiator?
    return K.function([input_img, K.learning_phase()], [saliency]) # K.learning_phase is ?

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instantiate a new model
        new_model = VGG19(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    t =  [l for l in model.layers[0].layers if l.name == layer_name]
    conv_output = t[0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def loadModel(inputFile):    
    model = load_model(inputFile)
    return model

def instantiateModel(cnn, inputWeights):
    cnnDict = {"vgg19": VGG19, "vgg16": VGG16, "xception":Xception, "resnet50": ResNet50,
                "inceptionV3": InceptionV3, "mobilenet": MobileNet}
    ##Initialize the neural network based on input string
    if (inputWeights == 'imagenet'):
        model =  cnnDict[cnn](weights=inputWeights)
    else:
        model = cnnDict[cnn](weights=None)

    return model
        

#TODO Add a static dictionary to support multiple cnn
def returnPredictionsandGenerateHeatMap(image, inputModel, numResults):
    preprocessed_input = load_image(image)
    predictions = inputModel.predict(preprocessed_input)  

    predicted_class = np.argmax(predictions)

    lastConvLayers = {"vgg19": 'block5_conv4', "vgg16": "block5_conv3"}

    #remove the heatmap files
    for entry in os.listdir("planck/static"): 
        fullPath = os.path.join("planck/static", entry)
        if os.path.isfile(fullPath):
            os.remove(fullPath)

    #heatmap only works for vgg nerual net architectures
    if ((inputModel.name == "vgg19") or (inputModel.name == "vgg16")):
        cam, heatmap = grad_cam(inputModel, preprocessed_input, predicted_class, lastConvLayers[inputModel.name])
        cv2.imwrite("planck/static/outputheatmap.jpg", cam)

        register_gradient()
        guided_model = modify_backprop(inputModel, 'GuidedBackProp')
        saliency_fn = compile_saliency_function(guided_model)
        saliency = saliency_fn([preprocessed_input, 0])
        gradcam = saliency[0] * heatmap[..., np.newaxis]
        cv2.imwrite("planck/static/guidedoutput.jpg", deprocess_image(gradcam))   
		 
    return decode_predictions(predictions, top=numResults)[0];

if __name__=="__main__":

    preprocessed_input = load_image(sys.argv[1])
    model = VGG19(weights='imagenet')
    print(model.summary())

    predictions = model.predict(preprocessed_input)
    print(predictions)
    top_1 = decode_predictions(predictions)[0][0]
    print('Predicted class:')
    print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

    '''predicted_class = np.argmax(predictions)
    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, final_conv_layer_name)
    cv2.imwrite("gradcam.jpg", cam)

    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp')
    saliency_fn = compile_saliency_function(guided_model)
    saliency = saliency_fn([preprocessed_input, 0])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))'''

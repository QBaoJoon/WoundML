from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage
from firstApp.postprocessing.hole_filling import fill_holes
from firstApp.postprocessing.remove_small_noise import remove_small_areas

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
from modelfunctions.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D

# openCV
import cv2
import numpy as np



img_height, img_width=224,224
threshold = 127
# with open('./models/imagenet_classes.json','r') as f:
#     labelInfo=f.read()

# labelInfo=json.loads(labelInfo)


model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model = Deeplabv3(input_shape=(224, 224, 3), classes=1)
        model.load_weights('models/0120-0.896442-0.814835_ft_wound.h5')



def index(request):
    context={'a':1}
    return render(request,'index.html',context)



def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    input_img='.'+filePathName
    img_ori = cv2.imread(input_img, 1)
    ori_shape = (img_ori.shape[1], img_ori.shape[0])
    img = cv2.resize(img_ori, (224, 224), interpolation=cv2.INTER_LINEAR)
    x=img/255.
    x = np.expand_dims(x, axis=0)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)
    predi = np.clip(predi, 0.0, 1.0) # 
    output_1c = np.uint8(predi[0] * 255.0)

    # write intermediate output image
    ext = '.' + input_img.split('.')[-1]
    out_ext = 'mask_'+ext
    output_path_mask = input_img.replace(ext, out_ext)

    output_1c = cv2.resize(output_1c, ori_shape, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path_mask, output_1c)

    output = cv2.imread(output_path_mask)
    _, threshed = cv2.threshold(output, threshold, 255, type=cv2.THRESH_BINARY)
    ################################################################################################################
    # call image post processing functions
    filled = fill_holes(threshed, threshold,0.1)
    denoised = remove_small_areas(filled, threshold, 0.05)

    denoised_overlay = cv2.addWeighted(img_ori, 0.6, np.uint8(denoised), 0.4, 0)
    # write output image and mask
    ext = '.' + input_img.split('.')[-1]
    out_ext = 'ovl_'+ext
    overlayed = input_img.replace(ext, out_ext)

    cv2.imwrite(output_path_mask, denoised) # mask filled
    cv2.imwrite(overlayed, denoised_overlay) # overlayed
    # if cv2.imwrite(input_img, img):
    context={'inputImage': input_img, 'mask':output_path_mask, 'overlayed':overlayed}
    return render(request,'run.html',context) 

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 
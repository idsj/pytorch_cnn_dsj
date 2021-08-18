import time
import torch
import os
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import Net


os.environ['CUDA_VISIBLE_DEVICES'] = "4"

root = "D:/workspace/pytorch_cnn_dsj/mnist/MNIST/raw/"

def cnn_recognition(image,model):
    image = image.convert('L')
    transformer = Net.resizeNormalize((28,28))
    image = transformer(image)
    image = image.view(1, *image.size())
   
    t_x = Variable(image)
    output = model(t_x)[0]
    pred = torch.max(output, 1)[1]
    print(pred.item())


if __name__ == '__main__':  
    cnn_model_path =  root + "log_CNN.pth"
    test_images_path = root + "test_images"
    print('loading pretrained model from {0}'.format(cnn_model_path))
    model = Net.CNN()
    model.load_state_dict(torch.load(cnn_model_path))
    model.eval()   
    print('loaded pretrained model from {0}'.format(cnn_model_path))

    files = sorted(os.listdir(test_images_path))
    for file in files:
        started = time.time()
        full_path = os.path.join(test_images_path, file)
        print("=============================================")
        print("ocr image is %s" % full_path)
        image = Image.open(full_path)

        cnn_recognition(image, model)
        finished = time.time()
        print('elapsed time: {0}'.format(finished - started))




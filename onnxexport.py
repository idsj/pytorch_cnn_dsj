import torch
import torch.onnx
import onnxruntime
import numpy as np
import onnx
from PIL import Image
import Net
from torch.autograd import Variable
import os
import time




def onnx_predict(image):
    onnx_file = onnx_model_new_path
    ort_session = onnxruntime.InferenceSession(onnx_file)
    input_name_detect = ort_session.get_inputs()[0].name  # 'data'
    # input.1
    print(input_name_detect)
    out_detect = ort_session.get_outputs()
    outputs_detect = list()
    for i in out_detect:
        outputs_detect.append(i.name)
    # ['268']
    print(outputs_detect)
    image = image.convert('L')
    transformer = Net.resizeNormalize((28, 28))
    image = transformer(image)
    image = image.view(1, *image.size())
    t_x = Variable(image)

    # 预测
    output = ort_session.run(["output1"], {"input1": t_x.numpy()}) 
    pred = np.argmax(output)
    print(pred)


root = "D:/workspace/pytorch_cnn_dsj/mnist/MNIST/raw/"
cnn_model_path =  root + "/model/cnn_model.pth"
test_images_path = root + "/test_images"
onnx_model_path = root + "/model/cnn_model.onnx"
onnx_model_new_path = root + "/model/cnn_model_new.onnx"
running_mode = 'gpu'


def pth_to_onnx():
    model = Net.CNN()
    if running_mode == 'gpu' and torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(cnn_model_path))
    else:
        model.load_state_dict(torch.load(cnn_model_path, map_location='cpu'))

    # data type nchw
    dummy_input = torch.randn(16, 1, 28, 28)
    input_names = ["input1"]
    output_names = ["output1"]
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, input_names=input_names,
                      output_names=output_names)


def onnx_change():
    model = onnx.load(onnx_model_path)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    onnx.save(model, onnx_model_new_path)


if __name__ == "__main__":
    pth_to_onnx()
    onnx_change()
    img1_file = "{0}/0.jpg".format(test_images_path)
    img1 = Image.open(img1_file)
    onnx_predict(img1)

    
    files = sorted(os.listdir(test_images_path))
    for file in files:
        started = time.time()
        full_path = os.path.join(test_images_path, file)
        print("=============================================")
        print("ocr image is %s" % full_path)
        image = Image.open(full_path)

        onnx_predict(image)
        finished = time.time()
        print('elapsed time: {0}'.format(finished - started))


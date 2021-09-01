import json
from commons import transform_image
import os
import onnxruntime


model_path = os.path.join('models', 'model_opti.onnx')
ort_session = onnxruntime.InferenceSession(model_path)
imagenet_class_index = json.load(open('imagenet_class_index.json'))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_prediction(image_bytes):
    try:
        img_y = transform_image(image_bytes)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = ort_outs[0]
    except Exception as e:
        print(e)
        return 404, 'error'
    return imagenet_class_index.get(str(outputs.argmax())), outputs.argmax()
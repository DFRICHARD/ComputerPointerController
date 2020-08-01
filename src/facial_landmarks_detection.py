import os
# import sys
# import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore

class FacialLandmarksDetection:
    '''
    Class for the Facial Landmarks Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_structure = model_name
        self.model_weights = os.path.splitext(model_name)[0] + ".bin"
        self.device = device

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError(
                "Could not Initialise the network. Do check if correct model path was entered")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        core = IECore()
        supported_layers = core.query_network(network=self.model, device_name="CPU")
        ### Check for any unsupported layers
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        self.exec_network = core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        input_name = self.input_name

        input_img = self.preprocess_input(image)

        input_dict = {input_name: input_img}

        infer_request_handle = self.exec_network.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name]
            l_eye, r_eye, eye_coords = self.preprocess_output(outputs, image)

        return l_eye, r_eye, eye_coords

    #def check_model(self):

    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        p_frame = cv2.resize(image, (w, h))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs, image):
        outputs = outputs[0]
        # left eye coordinates
        l_eye_xmin = int(outputs[0][0][0] * image.shape[1]) - 10
        l_eye_xmax = int(outputs[0][0][0] * image.shape[1]) + 10
        l_eye_ymin = int(outputs[1][0][0] * image.shape[0]) - 10
        l_eye_ymax = int(outputs[1][0][0] * image.shape[0]) + 10

        #right eye coordinates
        r_eye_xmin = int(outputs[2][0][0] * image.shape[1]) - 10
        r_eye_xmax = int(outputs[2][0][0] * image.shape[1]) + 10
        r_eye_ymin = int(outputs[3][0][0] * image.shape[0]) - 10
        r_eye_ymax = int(outputs[3][0][0] * image.shape[0]) + 10


        l_eye = image[l_eye_ymin:l_eye_ymax, l_eye_xmin:l_eye_xmax]
        r_eye = image[r_eye_ymin:r_eye_ymax, r_eye_xmin:r_eye_xmax]
        eye_coords = [[int(l_eye_xmin), int(l_eye_ymin), int(l_eye_xmax), int(l_eye_ymax)], [int(r_eye_xmin), int(r_eye_ymin), int(r_eye_xmax), int(r_eye_ymax)]]


        return l_eye, r_eye, eye_coords




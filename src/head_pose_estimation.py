'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class head_pose_estimation:
    '''
    Class for the head_pose_estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold= 0.6):
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.threshold = threshold

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        core = IECore()
        self.exec_network = core.load_network(network = self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        input_name = self.input_name

        input_img = self.preprocess_input(image)

        input_dict = {input_name: input_img}

        infer_request_handle = self.exec_network.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name]
            coords, image = self.draw_outputs(outputs, image)

            return coords, image


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        return image

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError

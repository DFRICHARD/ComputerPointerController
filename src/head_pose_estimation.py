import os
# import sys
# import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore

class HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_structure = model_name
        self.model_weights = os.path.splitext(model_name)[0] + ".bin"
        self.device = device

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Do check if correct model path was entered")

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
            outputs = infer_request_handle.outputs
            hp_angles = self.preprocess_output(outputs)

        return hp_angles

    #def check_model(self):

    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        p_frame = cv2.resize(image, (w, h))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs):

        y = outputs['angle_y_fc'][0][0]
        p = outputs['angle_p_fc'][0][0]
        r = outputs['angle_r_fc'][0][0]

        angles = [y, p, r]

        # if (display):
        #     out_image = self.draw_outputs(image,  angles, face_coords)
        return angles















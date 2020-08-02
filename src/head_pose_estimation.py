from math import cos, sin, pi
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

    def predict(self, image, face_crop, points, viz):
        input_name = self.input_name

        input_img = self.preprocess_input(face_crop)

        input_dict = {input_name: input_img}

        infer_request_handle = self.exec_network.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs
            out_image, hp_angles = self.preprocess_output(image, outputs, points, viz)

        return out_image, hp_angles

    #def check_model(self):

    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        p_frame = cv2.resize(image, (w, h))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, image, outputs, points, viz):

        y = outputs['angle_y_fc'][0][0]
        p = outputs['angle_p_fc'][0][0]
        r = outputs['angle_r_fc'][0][0]

        angles = [y, p, r]

        if (viz):
            # Face coordinates
            xmin = points[0]
            ymin = points[1]
            xmax = points[2]
            ymax = points[3]
            # I took the below code from here:
            # https://sudonull.com/post/6484-Intel-OpenVINO-on-Raspberry-Pi-2018-harvest
            cos_r = cos(r * pi / 180)
            sin_r = sin(r * pi / 180)
            sin_y = sin(y * pi / 180)
            cos_y = cos(y * pi / 180)
            sin_p = sin(p * pi / 180)
            cos_p = cos(p * pi / 180)

            x = int((xmin + xmax) / 2)
            y = int((ymin + ymax) / 2)

            # Center to right
            cv2.line(image, (x,y), (x+int(70*(cos_r*cos_y+sin_y*sin_p*sin_r)), y+int(70*cos_p*sin_r)), (128, 0, 128), 2)
            # Center to top
            cv2.line(image, (x, y), (x+int(70*(cos_r*sin_y*sin_p+cos_y*sin_r)), y-int(70*cos_p*cos_r)), (128, 0, 128), 2)
            # Center to forward
            cv2.line(image, (x, y), (x + int(70*sin_y*cos_p), y + int(70*sin_p)), (255, 0, 0), thickness=3)

        return image, angles















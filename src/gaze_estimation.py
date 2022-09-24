# import sys
import logging as log
import os
import cv2
import math
from openvino.inference_engine import IENetwork, IECore


class GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_structure = model_name
        self.model_weights = os.path.splitext(model_name)[0] + ".bin"
        self.device = device

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Do check if correct model path was entered")

        self.input_name = [i for i in self.model.inputs.keys()]
        self.output_name = [o for o in self.model.outputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        # self.output_shape = self.model.outputs[self.output_name].shape

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

    def predict(self, image, l_eye, r_eye, eye_coords, hp_angles, viz):
        processed_right_eye = self.preprocess_input(r_eye)
        processed_left_eye = self.preprocess_input(l_eye)
        input_dict = {'head_pose_angles': hp_angles,'left_eye_image': processed_left_eye, 'right_eye_image': processed_right_eye}

        infer_request_handle = self.exec_network.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name[0]]
            image_out, mouse_coord, gaze_vector = self.preprocess_output(image, outputs, hp_angles, eye_coords, viz)

        return image_out, mouse_coord, gaze_vector

    # def check_model(self):

    def preprocess_input(self, image):
        h, w = self.input_shape[2], self.input_shape[3]
        p_frame = cv2.resize(image, (w, h))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, image, outputs, hp_angles, eye_coords, viz):
        gaze_vector = outputs[0]
        angle_r_fc = hp_angles[2]

        cosine = math.cos(angle_r_fc * math.pi/180)
        sine = math.sin(angle_r_fc * math.pi/180)

        mouse_x = gaze_vector[0] * cosine + gaze_vector[1] * sine
        mouse_y = gaze_vector[0] * sine + gaze_vector[1] * cosine

        if (viz):
            l_eye_xcenter = int((eye_coords[0][0] + eye_coords[0][2])/2)
            l_eye_ycenter = int((eye_coords[0][1] + eye_coords[0][3])/2)

            r_eye_xcenter = int((eye_coords[1][0] + eye_coords[1][2])/2)
            r_eye_ycenter = int((eye_coords[1][1] + eye_coords[1][3])/2)

            cv2.arrowedLine(image, (l_eye_xcenter, l_eye_ycenter), (l_eye_xcenter + int(gaze_vector[0] * 150), l_eye_ycenter + int(-gaze_vector[1] * 150)), (0, 0, 0), 4)
            cv2.arrowedLine(image, (r_eye_xcenter, r_eye_ycenter), (r_eye_xcenter + int(gaze_vector[0] * 150), r_eye_ycenter + int(-gaze_vector[1] * 150)), (0, 0, 0), 4)



        return image, (mouse_x, mouse_y), gaze_vector

# import sys
# import logging as log
import os
import numpy as np
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore


class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions= None, threshold=0.6):
        self.model_structure = model_name
        self.model_weights = os.path.splitext(model_name)[0] + ".bin"
        self.device = device
        self.threshold = threshold
        self.extension = extensions


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

    def predict(self, image, viz):
        input_name = self.input_name

        input_img = self.preprocess_input(image)

        input_dict = {input_name: input_img}

        infer_request_handle = self.exec_network.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name]
            face_crop, points = self.preprocess_output(outputs, image, viz)
            # if len(points) == 0:
            #     log.error("No Face is detected...")
            #     return 0, 0, (0,0,0,0)

        return face_crop, points

    # def check_model(self):

    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        p_frame = cv2.resize(image, (w, h))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


    def preprocess_output(self, outputs, image, viz):

        points = []
        w, h = int(image.shape[1]), int(image.shape[0])
        frame_crop = image

        for box in outputs[0][0]:
            conf = box[2]
            if conf > self.threshold:
                xmin = int(box[3] * w)
                ymin = int(box[4] * h)
                xmax = int(box[5] * w)
                ymax = int(box[6] * h)
                points.append(xmin)
                points.append(ymin)
                points.append(xmax)
                points.append(ymax)

                frame_crop = image[ymin:ymax, xmin:xmax]

                if (viz):
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (128, 0, 128), 4)

        return frame_crop, points

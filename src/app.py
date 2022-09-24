import os
import logging as log
import cv2
import numpy as np
import time
from argparse import ArgumentParser


from mouse_controller import MouseController
from input_feeder import InputFeeder

from head_pose_estimation import HeadPoseEstimation
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation

logger = log.getLogger(__name__)
log.basicConfig(level=log.DEBUG)


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-fd", "--facedetection", required=True, type=str, help="Path to a face detection model .xml file.")

    parser.add_argument("-hpe", "--headpose", required=True, type=str,
                        help="Path to a head pose estimation model .xml file.")

    parser.add_argument("-fld", "--faciallandmark", required=True, type=str,
                        help="Path to a facial landmark detection .xml file.")

    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="Path to a gaze estimation model .xml file.")

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to input file (image or video file or type 'cam')")

    parser.add_argument("-ex", "--extension", required=False, type=str,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")

    parser.add_argument("-viz", "--display_flags", required=False, default = True, type=str,
                        help="Flag to display the outputs of the intermediate models")

    parser.add_argument("-d", "--device", required=False, type=str, default="CPU",
                        help="Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD"
                             "(CPU by default)")

    parser.add_argument("-pt_fd", "--fd_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
    return parser


def main():
    
    args = build_argparser().parse_args()


    Feed = None

    if args.input.lower() == "cam":
        Feed = InputFeeder("cam")
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        if not os.path.isfile(args.input):
            log.error("Unable to find specified input file")
            exit(1)
        Feed = InputFeeder("image", args.input)
    else:
        if not os.path.isfile(args.input):
            log.error("Unable to find specified video file")
            exit(1)
        Feed = InputFeeder("video", args.input)

    Feed.load_data()

    mc = MouseController('medium', 'medium')

    model_load_start = time.time()

    face_det = FaceDetection(args.facedetection, args.device, args.extension, args.fd_threshold)
    face_det.load_model()

    head_pose = HeadPoseEstimation(args.headpose, args.device, args.extension)
    head_pose.load_model()

    face_landmark = FacialLandmarksDetection(args.faciallandmark, args.device, args.extension)
    face_landmark.load_model()

    gaze_est = GazeEstimation(args.gazeestimation, args.device, args.extension)
    gaze_est.load_model()

    model_load_time = time.time() - model_load_start
    log.info('Model load time is: {} '.format(model_load_time))

    args.display_flags

    counter = 0

    for n, frame in Feed.next_batch():
        if frame is None:
            log.error('The input file is corrupted!!')
            exit()
        if not n:
            break

        image_copy = np.copy(frame)
        counter += 1

        key_pressed =cv2.waitKey(60)

        inference_time_start = time.time()
        face_crop, face_coords = face_det.predict(image_copy, args.display_flags)

        if len(face_coords) == 0:
            log.error("No face detected")
            continue

        image_out, hp_angles = head_pose.predict(image_copy, face_crop, face_coords, args.display_flags)
        image_out, l_eye, r_eye, eye_coords = face_landmark.predict(image_copy, face_crop, face_coords, args.display_flags)

        image_out, mouse_coords, gaze_vector = gaze_est.predict(image_out, l_eye, r_eye, eye_coords, hp_angles, args.display_flags)

        inference_time = time.time() - inference_time_start
        log.info("inference_time is: {} ".format(inference_time))

        cv2.imshow('frame', cv2.resize(image_out, (600, 400)))
        mc.move(mouse_coords[0], mouse_coords[1])

        if key_pressed == 27:
            break
    log.error("VideoStream  shutdown...")
    cv2.destroyAllWindows()
    Feed.close()


if __name__ == '__main__':
    main()

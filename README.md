# Computer Pointer Controller
This is a computer vision application making use of a blend of several computer vision models. This app enables controlling a mouse pointer through the movement of the head and eyes.
[Model Pipeline](./images/pipeline.png)

## Project Set Up and Installation
- Install Intel® Distribution of OpenVINO™ toolkit. Guide to install on your platform [HERE](https://docs.openvinotoolkit.org/latest/)

-  Create a virtual environment to work in and activate it:
  ```python3 -m venv cpcenv```
- cd to environment path and active it
  ```cpcenv\Scripts\activate.bat```
- set the openvino environment variable. (Since i'm using openvino 2020.1) 0n my windows it will be:
```C:\Program Files (x86)\IntelSWTools\openvino_2020.1.033\bin\setupvars.bat```
- Download models

1. [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
2. [Facial Landmarks Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
3. [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
4. [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)

- 

## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

We integrated the yolov8 example from the rknn_model_zoo based on https://github.com/leafqycc/rknn-multi-threaded, rewrote the post-processing function with GitHub Copilot, and removed the dependency on PyTorch.

# Introduction
* Utilizes multi-threaded asynchronous operations on the rknn model to increase the NPU utilization of RK3588/RK3588s, thereby improving inference frame rates (it should also work on devices like RK3568 after modification, but the author does not have an RK3568 development board).
* This branch uses the model [yolov5s_relu_tk2_RK3588_i8.rknn](https://github.com/airockchip/rknn_model_zoo), which replaces the silu activation function in the yolov5s model with relu, achieving significant performance improvement at the cost of slight accuracy loss. For more details, see [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo).
* The C++ implementation of this project can be found [here](https://github.com/leafqycc/rknn-cpp-Multithreading).

# Update Notes
* None

# Instructions
### Demonstration
* Clone the repository to your local machine, place the demo video from Releases in the root directory of the project, and run main.py to see the demo example.
* Switch to root user and run performance.sh to perform frequency operation (equivalent to enabling performance mode).
* Run rkcat.sh to check the current temperature and NPU utilization.

### Application Deployment
* Modify the modelPath in main.py to the path of your model.
* Modify the cap in main.py to the video/camera you want to run.
* Modify the TPEs in main.py to the number of threads you want, referring to the table below.
* Modify func.py to your required inference function, see the myFunc function for details.

# Multi-threaded Model Frame Rate Test
* Use performance.sh to perform CPU/NPU frequency setting to minimize error.
* The test model is [yolov5s_relu_tk2_RK3588_i8.rknn](https://github.com/airockchip/rknn_model_zoo).
* Test video can be found in Releases.

| Model\Threads | 1    |  2   | 3  |  4  | 5  | 6  |
| ----  | ----    | ----  |  ----  | ----  | ----  | ----  |
| yolov5s  | 27.4491 | 49.0747 | 65.3673  | 63.3204 | 71.8407 | 72.0590 |

# Additional Notes
* Under multi-threading, CPU and NPU usage are high, and **core temperature increases accordingly**, so please ensure proper cooling. It is recommended to use 1, 2, or 3 threads. Testing with a small copper piece for cooling showed temperatures of approximately 56°C, 64°C, and 69°C after running for three minutes.

# Acknowledgements
* https://github.com/ultralytics/yolov5
* https://github.com/rockchip-linux/rknn-toolkit2
* https://github.com/airockchip/rknn_model_zoo

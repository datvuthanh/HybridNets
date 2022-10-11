# -- C++ ROS Package for path planning, ONNXRuntime + TensorRT --
### This is tested on multiple Jetson devices (NX, AGX, OrinAGX)
### **WARNING**: Does not work out-of-the-box, you have to build ONNXRuntime-GPU.

# Install ROS
http://wiki.ros.org/ROS/Installation \
http://wiki.ros.org/catkin/Tutorials/create_a_workspace

# Copy this ROS package
```bash
cp -R . ~/catkin_ws/src/hybridnets_cpp/ 
```

# Prepare ONNX C++ library
```
├── onnx
│   ├── include
│   │   ├── onnxruntime_c_api.h
│   │   ├── onnxruntime_cxx_api.h
│   │   └── onnxruntime_cxx_inline.h
│   └── lib
│       ├── libonnxruntime_providers_cuda.so
│       ├── libonnxruntime_providers_shared.so
│       ├── libonnxruntime_providers_tensorrt.so
│       ├── libonnxruntime.so
│       └── libonnxruntime.so.1.11.0
```
You have to build all of these files :) \
It's not hard, it's just time consuming

# Build ONNXRuntime-GPU
For C++ api libraries and headers, you have to build from source on Jetson, good luck \
https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavier

**Important note:**
1. Clone a specific tag: https://onnxruntime.ai/docs/build/custom.html#version-of-onnx-runtime-to-build-from. \
Check your jetpack-tensorrt-onnxruntime version compability here: https://github.com/microsoft/onnxruntime/releases
2. Build onnx with --build_shared_lib: https://github.com/microsoft/onnxruntime/issues/9371

tl;dr on Jetson:
```bash
./build.sh --config Release --update --build --parallel --build_wheel --build_shared_lib --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --tensorrt_home /usr/lib/aarch64-linux-gnu --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="72"
```

When done, search for these files in particular and move them to /usr/local/lib/, or add them manually in ros' cmakelists.txt:

```
libonnxruntime_providers_cuda.so
libonnxruntime_providers_shared.so
libonnxruntime_providers_tensorrt.so
libonnxruntime.so
libonnxruntime.so.1.11.0 (change to your version)
```

tl;dr, with some edit:
```bash 
sudo find . -name "libonnxruntime*" 2>/dev/null
sudo cp $THE_ABOVE_FILES /usr/local/lib
```

c++ header files, include these in cmakelists:
```
onnxruntime_c_api.h
onnxruntime_cxx_api.h
onnxruntime_cxx_inline.h
```

tl;dr, with some edit:
```bash
sudo find . -name "onnxruntime_c*.h" 2>/dev/null
sudo cp $THE_ABOVE_FILES ~/catkin_ws/hybridnets_cpp/onnx/include/
```

# Prepare .onnx weight and .cpp anchor
After training, you have a .pth weight file. \
pth -> onnx, npy: use `python3 ../export.py` \
npy -> cpp: use `python3 extract_prior_box.py` \

Or for testing: \
Download example from this folder: https://drive.google.com/drive/folders/1dLgJ4LFutzaEi3s39cKP3CgCrF0UtAcF?usp=sharing \
`cp prior_bbox_256x384.cpp src/`

# Edit input topics
## Input: 1 RGB image (compressed or not)
```
// RAW
// image_transport::Subscriber sub = it.subscribe("/raw_image", 1, imageCallback);
// COMPRESSED
// image_transport::Subscriber sub = it.subscribe("/zed2i/zed_node/left_raw/image_raw_color", 1, imageCallback, ros::VoidPtr(), hints);
```
## Output: 
/road: grayscale image of pure road from network \
/lane: BEV image, road + object (no lane yet) \
/scan: laser_scan for planning \
/num_objects: number of objects

# Run
```
cd ~/catkin_ws
catkin_make
rosrun hybridnets_cpp hybridnets_cpp
```
First time takes ~1 hour to build .engine, then it caches for next times. \
If using rviz, edit `Fixed Frame` from `map` (default) to `odom` in order to visualize laser scan topic (or just change to whatever you want in the code).

# Debug
IF HAVING ISSUE WITH **OPENCV** (namely segmentation fault with basic functions like resize), you're probably having version conflict. I've not researched this further, apparently ROS is still using ROS's OPENCV even when I specify my exact OPENCV version in cmake. The imported version of `#include opencv` is my version too but it still fails. \
**SOLUTION**: Clone [`vision_opencv`](https://github.com/ros-perception/vision_opencv) in your `catkin_ws/src/` and build them (`catkin_make`) along this package. It magically corporates the correct OPENCV version somehow.

# Disclaimer
This is awfully hardcoded, so expect to change some strings and some numbers to make it work, or just open an issue lol

# Thanks to:
https://github.com/AbhiRP/Fake-LaserScan \
https://github.com/jdgalviss/autonomous_mobile_robot \
https://github.com/iwatake2222/play_with_tflite/tree/master/pj_tflite_perception_hybrid_nets \
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/276_HybridNets \
https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5

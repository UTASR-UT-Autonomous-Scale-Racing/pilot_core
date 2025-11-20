#include "python_controller.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>
#include <chrono>

PythonController::PythonController(const std::string& segmentation_config, const std::string& control_config)
    : segmentation_config_path_(segmentation_config)
    , control_config_path_(control_config)
    , segmentation_model_(nullptr)
    , control_model_(nullptr)
    , utils_module_(nullptr)
    , numpy_module_(nullptr)
    , should_stop_(false)
    , is_running_(false)
{
}

PythonController::~PythonController() {
    stop();
    finalizePython();
}

bool PythonController::initialize() {
    return initializePython();
}

bool PythonController::initializePython() {
    // Initialize Python interpreter
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            std::cerr << "Failed to initialize Python interpreter" << std::endl;
            return false;
        }
    }
    
    // Set up virtual environment path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.insert(0, '/home/zebrafish/pilot_core/venv/lib/python3.10/site-packages')");
    PyRun_SimpleString("sys.path.insert(0, '.')");
    
    // Initialize NumPy
    import_array1(false);
    
    // Import numpy
    numpy_module_ = PyImport_ImportModule("numpy");
    if (!numpy_module_) {
        std::cerr << "Failed to import numpy" << std::endl;
        PyErr_Print();
        return false;
    }
    
    // Import utils module
    utils_module_ = PyImport_ImportModule("utils");
    if (!utils_module_) {
        std::cerr << "Failed to import utils module" << std::endl;
        PyErr_Print();
        return false;
    }
    
    // Load segmentation model
    segmentation_model_ = loadModelFromConfig(segmentation_config_path_);
    if (!segmentation_model_) {
        std::cerr << "Failed to load segmentation model" << std::endl;
        return false;
    }
    
    // Load control model
    control_model_ = loadModelFromConfig(control_config_path_);
    if (!control_model_) {
        std::cerr << "Failed to load control model" << std::endl;
        return false;
    }
    
    std::cout << "Python controller initialized successfully" << std::endl;
    return true;
}

void PythonController::finalizePython() {
    Py_XDECREF(segmentation_model_);
    Py_XDECREF(control_model_);
    Py_XDECREF(utils_module_);
    Py_XDECREF(numpy_module_);
    
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
}

PyObject* PythonController::loadModelFromConfig(const std::string& config_path) {
    // Get the load_model_from_config function
    PyObject* load_func = PyObject_GetAttrString(utils_module_, "load_model_from_config");
    if (!load_func) {
        std::cerr << "Failed to get load_model_from_config function" << std::endl;
        PyErr_Print();
        return nullptr;
    }
    
    // Create arguments
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(config_path.c_str()));
    
    // Call the function
    PyObject* result = PyObject_CallObject(load_func, args);
    Py_DECREF(args);
    Py_DECREF(load_func);
    
    if (!result) {
        std::cerr << "Failed to load model from config: " << config_path << std::endl;
        PyErr_Print();
        return nullptr;
    }
    
    // Extract model from tuple (model, class_name)
    PyObject* model = PyTuple_GetItem(result, 0);
    Py_INCREF(model); // Increase reference count since we're returning it
    Py_DECREF(result);
    
    return model;
}

PyObject* PythonController::matToNumpyArray(const cv::Mat& mat) {
    int type = mat.type();
    npy_intp dims[3];
    int nd = 2;
    
    if (mat.channels() == 1) {
        dims[0] = mat.rows;
        dims[1] = mat.cols;
        nd = 2;
    } else {
        dims[0] = mat.rows;
        dims[1] = mat.cols;
        dims[2] = mat.channels();
        nd = 3;
    }
    
    int numpy_type;
    switch (type & CV_MAT_DEPTH_MASK) {
        case CV_8U: numpy_type = NPY_UINT8; break;
        case CV_32F: numpy_type = NPY_FLOAT32; break;
        case CV_16U: numpy_type = NPY_UINT16; break;
        default:
            std::cerr << "Unsupported OpenCV type: " << type << std::endl;
            return nullptr;
    }
    
    // Create numpy array
    PyObject* array = PyArray_SimpleNewFromData(nd, dims, numpy_type, mat.data);
    if (!array) {
        std::cerr << "Failed to create numpy array from cv::Mat" << std::endl;
        return nullptr;
    }
    
    // Make a copy to ensure data ownership
    PyObject* copy = PyArray_NewCopy((PyArrayObject*)array, NPY_CORDER);
    Py_DECREF(array);
    
    return copy;
}

void PythonController::start() {
    if (is_running_) {
        return;
    }
    
    should_stop_ = false;
    is_running_ = true;
    processing_thread_ = std::thread(&PythonController::processFrames, this);
    std::cout << "Python controller processing thread started" << std::endl;
}

void PythonController::stop() {
    if (!is_running_) {
        return;
    }
    
    should_stop_ = true;
    frame_available_.notify_all();
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    is_running_ = false;
    std::cout << "Python controller processing thread stopped" << std::endl;
}

void PythonController::submitFrame(const cv::Mat& rgbFrame, const cv::Mat& depthFrame, long long timestamp) {
    std::lock_guard<std::mutex> lock(frame_queue_mutex_);
    
    // Keep queue size reasonable
    while (frame_queue_.size() >= 5) {
        frame_queue_.pop();
    }
    
    frame_queue_.emplace(rgbFrame, depthFrame, timestamp);
    frame_available_.notify_one();
}

ControlOutput PythonController::getLatestControl() {
    std::lock_guard<std::mutex> lock(control_mutex_);
    return latest_control_;
}

void PythonController::processFrames() {
    std::cout << "Python controller processing thread running" << std::endl;
    
    while (!should_stop_) {
        FrameData frame;
        
        // Wait for frame or stop signal
        {
            std::unique_lock<std::mutex> lock(frame_queue_mutex_);
            frame_available_.wait(lock, [this]() { return !frame_queue_.empty() || should_stop_; });
            
            if (should_stop_) {
                break;
            }
            
            if (frame_queue_.empty()) {
                continue;
            }
            
            frame = frame_queue_.front();
            frame_queue_.pop();
        }
        
        try {
            // Convert frames to numpy arrays
            PyObject* depth_array = matToNumpyArray(frame.depthFrame);
            PyObject* rgb_array = matToNumpyArray(frame.rgbFrame);
            
            if (!depth_array || !rgb_array) {
                std::cerr << "Failed to convert frames to numpy arrays" << std::endl;
                Py_XDECREF(depth_array);
                Py_XDECREF(rgb_array);
                continue;
            }
            
            // Call segmentation model get_mask method
            PyObject* get_mask_method = PyObject_GetAttrString(segmentation_model_, "get_mask");
            if (!get_mask_method) {
                std::cerr << "Failed to get get_mask method from segmentation model" << std::endl;
                PyErr_Print();
                Py_DECREF(depth_array);
                Py_DECREF(rgb_array);
                continue;
            }
            
            PyObject* mask_args = PyTuple_New(2);
            PyTuple_SetItem(mask_args, 0, depth_array); // depth_frame first
            PyTuple_SetItem(mask_args, 1, rgb_array);   // rgb_frame second
            
            PyObject* segments = PyObject_CallObject(get_mask_method, mask_args);
            Py_DECREF(mask_args);
            Py_DECREF(get_mask_method);
            
            if (!segments) {
                std::cerr << "Failed to get segments from segmentation model" << std::endl;
                PyErr_Print();
                continue;
            }
            
            // Create dummy sensor data (quaternion + accelerometer)
            std::vector<float> sensors = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 9.81f}; // w,x,y,z,ax,ay,az
            
            // Call control model
            ControlOutput control = callControlModel(sensors, segments, frame.timestamp);
            
            Py_DECREF(segments);
            
            // Update latest control output
            {
                std::lock_guard<std::mutex> lock(control_mutex_);
                latest_control_ = control;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Exception in processFrames: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in processFrames" << std::endl;
        }
    }
    
    std::cout << "Python controller processing thread exiting" << std::endl;
}

ControlOutput PythonController::callControlModel(const std::vector<float>& sensors, PyObject* segments, long long timestamp) {
    // Get the get_action method
    PyObject* get_action_method = PyObject_GetAttrString(control_model_, "get_action");
    if (!get_action_method) {
        std::cerr << "Failed to get get_action method from control model" << std::endl;
        PyErr_Print();
        return ControlOutput();
    }
    
    // Create sensor array
    npy_intp sensor_dims[] = {static_cast<npy_intp>(sensors.size())};
    PyObject* sensor_array = PyArray_SimpleNew(1, sensor_dims, NPY_FLOAT32);
    if (!sensor_array) {
        std::cerr << "Failed to create sensor array" << std::endl;
        Py_DECREF(get_action_method);
        return ControlOutput();
    }
    
    float* sensor_data = (float*)PyArray_DATA((PyArrayObject*)sensor_array);
    for (size_t i = 0; i < sensors.size(); ++i) {
        sensor_data[i] = sensors[i];
    }
    
    // Create timestamp object
    PyObject* timestamp_obj = PyFloat_FromDouble(static_cast<double>(timestamp));
    
    // Call get_action with appropriate arguments
    // Try different argument patterns based on different model signatures
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, sensor_array);
    PyTuple_SetItem(args, 1, segments);
    Py_INCREF(segments); // Increase ref count since PyTuple_SetItem steals reference
    PyTuple_SetItem(args, 2, timestamp_obj);
    
    PyObject* result = PyObject_CallObject(get_action_method, args);
    Py_DECREF(args);
    Py_DECREF(get_action_method);
    
    if (!result) {
        std::cerr << "Failed to call control model get_action" << std::endl;
        PyErr_Print();
        return ControlOutput();
    }
    
    // Extract steering and throttle from result tuple
    if (!PyTuple_Check(result) || PyTuple_Size(result) != 2) {
        std::cerr << "Invalid result from control model" << std::endl;
        Py_DECREF(result);
        return ControlOutput();
    }
    
    float steering = static_cast<float>(PyFloat_AsDouble(PyTuple_GetItem(result, 0)));
    float throttle = static_cast<float>(PyFloat_AsDouble(PyTuple_GetItem(result, 1)));
    
    Py_DECREF(result);
    
    return ControlOutput(steering, throttle);
}
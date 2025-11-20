#pragma once

#include <Python.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>

struct FrameData {
    cv::Mat rgbFrame;
    cv::Mat depthFrame;
    long long timestamp;
    
    FrameData() = default;
    FrameData(const cv::Mat& rgb, const cv::Mat& depth, long long ts)
        : rgbFrame(rgb.clone()), depthFrame(depth.clone()), timestamp(ts) {}
};

struct ControlOutput {
    float steering;
    float throttle;
    bool valid;
    
    ControlOutput() : steering(0.5f), throttle(0.5f), valid(false) {}
    ControlOutput(float s, float t) : steering(s), throttle(t), valid(true) {}
};

class PythonController {
public:
    PythonController(const std::string& segmentation_config, const std::string& control_config);
    ~PythonController();
    
    bool initialize();
    void start();
    void stop();
    void submitFrame(const cv::Mat& rgbFrame, const cv::Mat& depthFrame, long long timestamp);
    ControlOutput getLatestControl();
    
private:
    std::string segmentation_config_path_;
    std::string control_config_path_;
    
    // Python objects
    PyObject* segmentation_model_;
    PyObject* control_model_;
    PyObject* utils_module_;
    PyObject* numpy_module_;
    
    // Threading
    std::thread processing_thread_;
    std::atomic<bool> should_stop_;
    std::atomic<bool> is_running_;
    
    // Frame queue
    std::queue<FrameData> frame_queue_;
    std::mutex frame_queue_mutex_;
    std::condition_variable frame_available_;
    
    // Control output
    ControlOutput latest_control_;
    std::mutex control_mutex_;
    
    // Processing methods
    void processFrames();
    PyObject* loadModelFromConfig(const std::string& config_path);
    PyObject* matToNumpyArray(const cv::Mat& mat);
    ControlOutput callControlModel(const std::vector<float>& sensors, PyObject* segments, long long timestamp);
    
    // Python utilities
    bool initializePython();
    void finalizePython();
};
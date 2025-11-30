#include <cstdio>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <cmath>
#include <sys/stat.h>
#include <iomanip>
#include <string>
#include <getopt.h>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include "System.h"
#include "MapPoint.h"
#include "util.h"
#include "python_controller.h"

#include "utility.hpp"
#include "depthai/depthai.hpp"
#include "depthai/utility/Clock.hpp"

static std::atomic<bool> downscaleColor{true};
static std::atomic<bool> display{false};
static std::atomic<bool> record{false};
static constexpr int fps = 120;
static constexpr auto monoRes = dai::MonoCameraProperties::SensorResolution::THE_480_P;

// WLS parameters, taken from the OpenCV WLS filter docs recommended values.
#define WLS_LAMBDA (8000)
#define WLS_SIGMA (1.2)


// Recording support
static std::string recordPath;

// Save RGBD frame to binary file: CV_32FC4 (R,G,B,D)
void saveRgbdFrame(const cv::Mat rgbFrame, cv::Mat depthFrame, long long ms, const std::string path) {
    // Convert to float types properly (create new matrices)
    cv::Mat rgbF, depthF;
    rgbFrame.convertTo(rgbF, CV_32F, 1.0/255.0); // Normalize RGB to 0-1 range
    depthFrame.convertTo(depthF, CV_32F);

    // Split RGB into channels
    std::vector<cv::Mat> rgbChannels;
    cv::split(rgbF, rgbChannels);
    
    // Add depth channel directly
    rgbChannels.push_back(depthF);
    
    // Merge into RGBD
    cv::Mat rgbd;
    cv::merge(rgbChannels, rgbd); // CV_32FC4
    std::string file = path + "/rgbd_frame_" + std::to_string(ms) + ".bin";

    std::ofstream ofs(file, std::ios::binary);
    if(!ofs) {
        std::cerr << "Failed to write: " << file << std::endl;
        return;
    }

    int width = rgbd.cols;
    int height = rgbd.rows;
    int channels = rgbd.channels();
    int type = rgbd.type();

    // Write metadata
    ofs.write((char*)&width, sizeof(width));
    ofs.write((char*)&height, sizeof(height));
    ofs.write((char*)&channels, sizeof(channels));
    ofs.write((char*)&type, sizeof(type));

    // Write frame data
    ofs.write((char*)rgbd.data, rgbd.total() * rgbd.elemSize());
    
    if(ofs.good()) {
        std::cout << "Saved frame: " << file << std::endl;
    } else {
        std::cerr << "Error writing frame: " << file << std::endl;
    }
}

// Append pose together with newly observed map/key points for a timestamp.
void savePoseAndPoints(
    std::vector<cv::Point3f> mapPoints,
    std::vector<cv::KeyPoint> keypoints,
    cv::Mat currPose,
    long long unix_timestamp_ms,
    std::string path
) {
    if(currPose.empty()) {
        return;
    }

    cv::Mat poseFloat;
    currPose.convertTo(poseFloat, CV_32F);
    if(!poseFloat.isContinuous()) {
        poseFloat = poseFloat.clone();
    }

    const size_t poseValueCount = poseFloat.total() * poseFloat.channels();
    std::vector<float> poseBuffer;
    poseBuffer.assign(poseFloat.ptr<float>(0), poseFloat.ptr<float>(0) + poseValueCount);

    std::vector<float> mapPointBuffer;
    mapPointBuffer.reserve(mapPoints.size() * 3);
    for(const auto& mapPoint : mapPoints) {
        mapPointBuffer.push_back(mapPoint.x);
        mapPointBuffer.push_back(mapPoint.y);
        mapPointBuffer.push_back(mapPoint.z);
    }
    const int32_t validMapPointCount = static_cast<int32_t>(mapPointBuffer.size() / 3);

    struct KeyPointRecord {
        float x;
        float y;
        float size;
        float angle;
        float response;
        int32_t octave;
        int32_t class_id;
    };

    std::vector<KeyPointRecord> keypointBuffer;
    keypointBuffer.reserve(keypoints.size());
    for(const auto& kp : keypoints) {
        keypointBuffer.push_back(
            KeyPointRecord{kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id}
        );
    }
    const int32_t keypointCount = static_cast<int32_t>(keypointBuffer.size());

    static std::mutex fileMutex;
    static std::string activeRecordRoot;
    static std::string mapPointsFilePath;
    static bool fileInitialized = false;
    static long long fileEpochMs = 0;

    std::lock_guard<std::mutex> guard(fileMutex);

    if(!fileInitialized || activeRecordRoot != path) {
        fileEpochMs = unix_timestamp_ms;
        mapPointsFilePath = path + "/map_points_" + std::to_string(fileEpochMs);
        std::ofstream resetStream(mapPointsFilePath, std::ios::binary | std::ios::trunc);
        if(!resetStream) {
            std::cerr << "Failed to create map points file: " << mapPointsFilePath << std::endl;
            fileInitialized = false;
            return;
        }
        activeRecordRoot = path;
        fileInitialized = true;
    }

    std::ofstream ofs(mapPointsFilePath, std::ios::binary | std::ios::app);
    if(!ofs) {
        std::cerr << "Failed to open map points file for appending: " << mapPointsFilePath << std::endl;
        return;
    }

    const int32_t poseRows = poseFloat.rows;
    const int32_t poseCols = poseFloat.cols;
    const int32_t poseType = poseFloat.type();

    ofs.write(reinterpret_cast<const char*>(&unix_timestamp_ms), sizeof(unix_timestamp_ms));
    ofs.write(reinterpret_cast<const char*>(&poseRows), sizeof(poseRows));
    ofs.write(reinterpret_cast<const char*>(&poseCols), sizeof(poseCols));
    ofs.write(reinterpret_cast<const char*>(&poseType), sizeof(poseType));
    ofs.write(reinterpret_cast<const char*>(poseBuffer.data()), poseBuffer.size() * sizeof(float));

    ofs.write(reinterpret_cast<const char*>(&validMapPointCount), sizeof(validMapPointCount));
    if(validMapPointCount > 0) {
        ofs.write(reinterpret_cast<const char*>(mapPointBuffer.data()), mapPointBuffer.size() * sizeof(float));
    }

    ofs.write(reinterpret_cast<const char*>(&keypointCount), sizeof(keypointCount));
    if(keypointCount > 0) {
        ofs.write(reinterpret_cast<const char*>(keypointBuffer.data()), keypointBuffer.size() * sizeof(KeyPointRecord));
    }

    if(!ofs.good()) {
        std::cerr << "Error writing pose/map data: " << mapPointsFilePath << std::endl;
    }
}

// Optional display blend weights
static float rgbWeight = 0.4f;
static float depthWeight = 0.6f;

static void updateBlendWeights(int percentRgb, void* ctx) {
    rgbWeight = float(percentRgb) / 100.f;
    depthWeight = 1.f - rgbWeight;
}

int main(int argc, char** argv) {
    using namespace std;

    string vocabulary_path;
    string settings_path;
    string segmentation_config_path;
    string control_config_path;
    string file_name;
    bool bFileName = false;
    bool use_python_models = false;

    // Parse command line arguments
    static struct option long_options[] = {
        {"display", no_argument, 0, 'd'},
        {"record", required_argument, 0, 'r'},
        {"segmentation-config", required_argument, 0, 's'},
        {"control-config", required_argument, 0, 'c'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "dr:s:c:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'd':
                display = true;
                cout << "Display enabled" << endl;
                break;
            case 'r':
                record = true;
                recordPath = string(optarg);
                cout << "Recording enabled, path: " << recordPath << endl;
                break;
            case 's':
                segmentation_config_path = string(optarg);
                use_python_models = true;
                cout << "Segmentation config: " << segmentation_config_path << endl;
                break;
            case 'c':
                control_config_path = string(optarg);
                cout << "Control config: " << control_config_path << endl;
                break;
            case 'h':
                cout << "Usage: " << argv[0] << " [OPTIONS] VOCAB_PATH SETTINGS_PATH [OUTPUT_FILE]" << endl;
                cout << "Options:" << endl;
                cout << "  --display, -d                    Enable frame display" << endl;
                cout << "  --record PATH, -r PATH           Enable recording to specified directory" << endl;
                cout << "  --segmentation-config, -s PATH   Path to segmentation model config" << endl;
                cout << "  --control-config, -c PATH        Path to control model config" << endl;
                cout << "  --help, -h                       Show this help message" << endl;
                cout << endl;
                cout << "Arguments:" << endl;
                cout << "  VOCAB_PATH            Path to ORB vocabulary file" << endl;
                cout << "  SETTINGS_PATH         Path to camera settings file" << endl;
                cout << "  OUTPUT_FILE           Optional output filename" << endl;
                return 0;
            case '?':
                cerr << "Unknown option. Use --help for usage information." << endl;
                return 1;
            default:
                abort();
        }
    }

    // Parse remaining positional arguments
    int remaining_args = argc - optind;
    if (remaining_args < 2) {
        cerr << "Error: Missing required arguments." << endl;
        cerr << "Usage: " << argv[0] << " [OPTIONS] VOCAB_PATH SETTINGS_PATH [OUTPUT_FILE]" << endl;
        cerr << "Use --help for more information." << endl;
        return 1;
    }

    vocabulary_path = string(argv[optind]);
    settings_path = string(argv[optind + 1]);
    
    if (remaining_args >= 3) {
        file_name = string(argv[optind + 2]);
        bFileName = true;
    }

    // Validate Python model configurations if using them
    if (use_python_models) {
        if (segmentation_config_path.empty() || control_config_path.empty()) {
            cerr << "Error: Both segmentation and control config paths are required when using Python models." << endl;
            return 1;
        }
        cout << "Python models enabled" << endl;
        cout << "Segmentation config: " << segmentation_config_path << endl;
        cout << "Control config: " << control_config_path << endl;
    }

    // Verify recording directory exists if recording was enabled
    if(record) {
        // Check if the directory exists
        struct stat info;
        if(stat(recordPath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            std::cerr << "Error: Recording directory does not exist: " << recordPath << std::endl;
            return 1;
        }
        std::cout << "Recording verified. Directory exists: " << recordPath << std::endl;
    }

    // Create pipeline
    dai::Pipeline pipeline;
    dai::Device device;

    // Define sources and outputs
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto left = pipeline.create<dai::node::MonoCamera>();
    auto right = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto sync = pipeline.create<dai::node::Sync>();
    auto xoutGrp = pipeline.create<dai::node::XLinkOut>();
    xoutGrp->setStreamName("xout");

    pipeline.setXLinkChunkSize(0);
    sync->setSyncThreshold(std::chrono::milliseconds(50));
    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setFps(fps);
    if(downscaleColor) camRgb->setPreviewSize(640, 480);

    try {
        auto calibData = device.readCalibration2();
        auto lensPosition = calibData.getLensPosition(dai::CameraBoardSocket::CAM_A);
        if(lensPosition) {
            camRgb->initialControl.setManualFocus(lensPosition);
        }
    } catch(const std::exception& ex) {
        std::cout << ex.what() << std::endl;
        return 1;
    }

    left->setResolution(monoRes);
    left->setCamera("left");
    left->setFps(fps);
    right->setResolution(monoRes);
    right->setCamera("right");
    right->setFps(fps);

    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::DEFAULT);
    stereo->setLeftRightCheck(true);
    stereo->setSubpixel(true);
    stereo->setExtendedDisparity(false);
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);
    stereo->setOutputSize(640, 480);

    auto rgbFrameName = "rgb";
    auto depthFrameName = "depth";
    left->out.link(stereo->left);
    right->out.link(stereo->right);
    stereo->disparity.link(sync->inputs[depthFrameName]);
    camRgb->preview.link(sync->inputs[rgbFrameName]);
    sync->out.link(xoutGrp->input);

    // Connect to device and start pipeline
    std::cout << "DEBUG: Starting pipeline on device..." << std::endl;
    try {
        device.startPipeline(pipeline);
        std::cout << "DEBUG: Pipeline started successfully" << std::endl;
    } catch(const std::exception& ex) {
        std::cout << "DEBUG: Exception starting pipeline: " << ex.what() << std::endl;
        return 1;
    }
    
    auto queue = device.getOutputQueue("xout", 10, true);

    float disparityMultiplier = 255 / stereo->initialConfig.getMaxDisparity();

    auto wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
    wls_filter->setLambda(WLS_LAMBDA);
    wls_filter->setSigmaColor(WLS_SIGMA);


    // Initialize Python controller if enabled
    std::unique_ptr<PythonController> python_controller;
    std::atomic<bool> python_thread_running{false};
    std::thread python_thread;
    
    if (use_python_models) {
        python_controller = std::make_unique<PythonController>(segmentation_config_path, control_config_path);
        if (!python_controller->initialize()) {
            std::cerr << "Failed to initialize Python controller" << std::endl;
            return 1;
        }
        python_controller->start();
        
        // Start Python controller thread for processing
        python_thread_running = true;
        python_thread = std::thread([&python_controller, &python_thread_running]() {
            while (python_thread_running) {
                // Get latest control output periodically
                ControlOutput control = python_controller->getLatestControl();
                if (control.valid) {
                    std::cout << "Control output - Steering: " << control.steering 
                              << ", Throttle: " << control.throttle << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 100Hz control output rate
            }
        });
        
        std::cout << "Python controller started successfully" << std::endl;
    }

    ORB_SLAM3::System SLAM(
        vocabulary_path.c_str(), settings_path.c_str(), 
        ORB_SLAM3::System::RGBD, 
        display ? true : false, 0, file_name
    );

    // Formatter, for printing out matrices in a reasonable way.
    cv::Ptr<cv::Formatter> fmt = cv::Formatter::get(cv::Formatter::FMT_DEFAULT);
    fmt->set64fPrecision(3);
    fmt->set32fPrecision(3);

    // The time of each frame is required for SLAM, so we take an epoch time
    // (i.e. our start time) now
    auto slam_epoch = std::chrono::steady_clock::now();


    while(true) {
        auto msgGrp = queue->get<dai::MessageGroup>();

        cv::Mat latestRgbFrame;
        cv::Mat latestDepthFrame;

        for(auto& frame : *msgGrp) {
            auto imgFrame = std::dynamic_pointer_cast<dai::ImgFrame>(frame.second);
            if(!imgFrame) {
                continue;
            }
            
            if(frame.first == depthFrameName) {
                try {
                    if(imgFrame) {
                        auto data = imgFrame->getData();
                        auto width = imgFrame->getWidth();
                        auto height = imgFrame->getHeight();
                        if(data.size() == width * height * 2) {  // 2 bytes per pixel for 16-bit depth
                            latestDepthFrame = cv::Mat(height, width, CV_16UC1, data.data()).clone();
                        }
                    }
                } catch(const std::exception& ex) {
                    // Handle exception if needed
                } catch(...) {
                    // Handle unknown exception if needed
                }
            } else if(frame.first == rgbFrameName) {
                try {
                    if(imgFrame) {
                        auto data = imgFrame->getData();
                        auto width = imgFrame->getWidth();
                        auto height = imgFrame->getHeight();
                        if(data.size() == width * height * 3) {
                            latestRgbFrame = cv::Mat(height, width, CV_8UC3, data.data()).clone();
                        }
                    }
                } catch(const std::exception& ex) {
                    // Handle exception if needed
                } catch(...) {
                    // Handle unknown exception if needed
                }
            }
        }

        if(!latestRgbFrame.empty() && !latestDepthFrame.empty()) {

            //auto elapsed_time = std::chrono::steady_clock::now() - slam_epoch;
            //double frame_timestamp_s = elapsed_time.count() / 1000000000.0;

            // Get current Unix timestamp in nanoseconds
            auto now = std::chrono::system_clock::now();
            auto unix_timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

            // Submit frames to Python controller if enabled (non-blocking)
            if (use_python_models && python_controller) {
                // Submit frame asynchronously - the python controller thread handles processing
                python_controller->submitFrame(latestRgbFrame, latestDepthFrame, unix_timestamp_ms);
            }

            // Record if enabled
            if(record) {            
                std::thread saverThread(saveRgbdFrame, latestRgbFrame.clone(), latestDepthFrame.clone(), unix_timestamp_ms, recordPath);
                saverThread.detach();
            }

            // Optional display
            if(display) {
                cv::Mat depthBgr;
                latestDepthFrame.convertTo(depthBgr, CV_8UC1, disparityMultiplier);
                cv::applyColorMap(depthBgr, depthBgr, cv::COLORMAP_JET);
                if(depthBgr.channels() < 3) {
                    cv::cvtColor(depthBgr, depthBgr, cv::COLOR_GRAY2BGR);
                }
                cv::Mat blended;
                cv::addWeighted(latestRgbFrame, rgbWeight, depthBgr, depthWeight, 0, blended);
                cv::imshow("rgb-depth", blended);
            }

            // Pass the images into the SLAM system.
            SLAM.TrackRGBD(
                latestRgbFrame,
                latestDepthFrame,
                unix_timestamp_ms
            );

            // Clear frame buffers after SLAM processing
            latestRgbFrame.release();
            latestDepthFrame.release();

            ORB_SLAM3::System::TrajectorySample trajectorySample;
            bool trajectoryAvailable = SLAM.getCurrentTrajectory(trajectorySample);
            ORB_SLAM3::System::KeyFrameTrajectorySample keyframeSample;
            SLAM.getCurrentKeyFrameTrajectory(keyframeSample);

            if(record && trajectoryAvailable) {
                cv::Mat poseMat(4, 4, CV_32F);
                Eigen::Matrix4f eigen_mat = trajectorySample.pose.matrix();
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        poseMat.at<float>(i, j) = eigen_mat(i, j);
                    }
                }

                std::vector<cv::Point3f> mapPointPositions;
                mapPointPositions.reserve(trajectorySample.newMapPoints.size());
                for(auto* mapPoint : trajectorySample.newMapPoints) {
                    if(!mapPoint || mapPoint->isBad()) {
                        continue;
                    }
                    const auto worldPos = mapPoint->GetWorldPos();
                    mapPointPositions.emplace_back(worldPos.x(), worldPos.y(), worldPos.z());
                }

                long long poseTimestamp = static_cast<long long>(std::llround(trajectorySample.timestamp));

                std::thread saverThread(
                    savePoseAndPoints,
                    mapPointPositions,
                    trajectorySample.keypoints,
                    poseMat,
                    poseTimestamp,
                    recordPath
                );
                saverThread.detach();
            }

        }
    }

    // Stop all SLAM threads
    SLAM.Shutdown();

    // Stop Python controller if running
    if (use_python_models && python_controller) {
        // Stop the Python controller thread
        python_thread_running = false;
        if (python_thread.joinable()) {
            python_thread.join();
        }
        
        python_controller->stop();
        std::cout << "Python controller stopped" << std::endl;
    }

    return EXIT_SUCCESS;
}

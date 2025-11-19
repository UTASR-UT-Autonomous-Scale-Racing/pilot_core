#include <cstdio>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <atomic>
#include <sys/stat.h>
#include <iomanip>
#include <string>
#include <getopt.h>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include "System.h"
#include "util.h"

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
void saveRgbdFrame(const cv::Mat rgbFrame, cv::Mat depthFrame, long long ns, const std::string path) {

    cv::Mat depthF;
    depthFrame.convertTo(depthF, CV_32F);

    cv::Mat rgbF;
    rgbFrame.convertTo(rgbF, CV_32F);

    std::vector<cv::Mat> rgbSplit;
    cv::split(rgbF, rgbSplit);

    std::vector<cv::Mat> ch;
    ch.insert(ch.end(), rgbSplit.begin(), rgbSplit.end());
    ch.push_back(depthF);

    cv::Mat rgbd;
    cv::merge(ch, rgbd); // CV_32FC4

    std::string file = path + "/frame_" + std::to_string(ns) + ".bin";

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

// Optional display blend weights
static float rgbWeight = 0.4f;
static float depthWeight = 0.6f;

static void updateBlendWeights(int percentRgb, void* ctx) {
    rgbWeight = float(percentRgb) / 100.f;
    depthWeight = 1.f - rgbWeight;
}

typedef struct _Pose {
    cv::Mat position;
    cv::Mat rotation;
} Pose;

int main(int argc, char** argv) {
    using namespace std;

    string vocabulary_path;
    string settings_path;
    string file_name;
    bool bFileName = false;

    // Parse command line arguments
    static struct option long_options[] = {
        {"display", no_argument, 0, 'd'},
        {"record", required_argument, 0, 'r'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "dr:h", long_options, &option_index)) != -1) {
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
            case 'h':
                cout << "Usage: " << argv[0] << " [OPTIONS] VOCAB_PATH SETTINGS_PATH [OUTPUT_FILE]" << endl;
                cout << "Options:" << endl;
                cout << "  --display, -d          Enable frame display" << endl;
                cout << "  --record PATH, -r PATH Enable recording to specified directory" << endl;
                cout << "  --help, -h             Show this help message" << endl;
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


    cv::Mat R1, R2, P1, P2, Q;

    ORB_SLAM3::System SLAM(
        vocabulary_path.c_str(), settings_path.c_str(), 
        ORB_SLAM3::System::RGBD, 
        display ? true : false, 0, file_name
    );

    // Formatter, for printing out matrices in a reasonable way.
    cv::Ptr<cv::Formatter> fmt = cv::Formatter::get(cv::Formatter::FMT_DEFAULT);
    fmt->set64fPrecision(3);
    fmt->set32fPrecision(3);

    // We also want somewhere to store our pose data
    Pose pose;

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

            auto elapsed_time = std::chrono::steady_clock::now() - slam_epoch;
            double frame_timestamp_s = elapsed_time.count() / 1000000000.0;

            std::cout << std::setprecision(4) << frame_timestamp_s << ": ";

            // Record if enabled
            if(record) {
                cv::Mat latestDepthFrameCopy = latestDepthFrame.clone();
                cv::Mat latestRgbFrameCopy = latestRgbFrame.clone();

                std::thread saverThread(saveRgbdFrame, latestDepthFrameCopy, latestRgbFrameCopy, frame_timestamp_s, recordPath);
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

            // Pass the images into the SLAM system. This produces a Sophus pose
            Sophus::SE3f pose_sophus = SLAM.TrackRGBD(
                latestRgbFrame,
                latestDepthFrame,
                frame_timestamp_s
            );

            // Convert Sophus SE3f to cv::Mat (4x4 transformation matrix)
            cv::Mat raw_pose;
            if (!pose_sophus.matrix().isZero()) {
                // Convert Sophus SE3f matrix to cv::Mat
                Eigen::Matrix4f eigen_mat = pose_sophus.matrix();
                raw_pose = cv::Mat(4, 4, CV_32F);
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        raw_pose.at<float>(i, j) = eigen_mat(i, j);
                    }
                }
            }

            // The output pose may be empty if the system was unable to track the
            // movement, so only get position and rotation if pose isn't empty. We
            // also put this info an a localisation fix available flag for later
            // use. 
            bool loc_fix_available = !raw_pose.empty();
            if (loc_fix_available) {
                // The pose matrix is a 4x4 extrinsic matrix, with the form:
                // [R_3x3 T_3x1; [0 0 0 1]], we can find the camera position with 
                // C = -R'T (R' = R transpose).
                pose.rotation = raw_pose(cv::Rect(0, 0, 3, 3));
                cv::Mat T = raw_pose(cv::Rect(3, 0, 1, 3));
                pose.position = -pose.rotation.t()*T;

                // Print the updated position, but transpose it so that instead of
                // a column vector we have a row vector, which is easier to read.
                std::cout << 
                    "position: " << 
                    fmt->format(pose.position.t()) << 
                    std::endl;
            }
            else {
                // If we didn't get a pose update log it.
                std::cout << "no pose update" << std::endl;
            }

            // Continue processing frames (removed display code for headless operation)
        }
    }

    // Stop all SLAM threads
    SLAM.Shutdown();

    return EXIT_SUCCESS;
}

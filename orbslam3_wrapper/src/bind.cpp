/**
 * pybind11 wrapper for ORB_SLAM3 — exposes a minimal Python API for the
 * pilot_core multiprocessing pipeline.
 *
 * Exposed API:
 *   orbslam3_python.System(vocab, settings, sensor_type)
 *       .track_rgbd(rgb, depth, timestamp) -> np.ndarray (4x4 float32)
 *       .map_changed()                     -> bool
 *       .get_all_map_points()              -> np.ndarray (N x 3 float32)
 *       .get_all_keyframe_poses()          -> np.ndarray (K x 4 x 4 float32)
 *       .get_trajectory()                  -> list of np.ndarray (4x4 float32)
 *       .is_loop_detected()                -> bool
 *       .activate_localization_mode()      -> None
 *       .deactivate_localization_mode()    -> None
 *       .get_tracking_state()              -> int
 *       .shutdown()                        -> None
 *
 *   Constants:
 *       orbslam3_python.RGBD  = 2
 *       orbslam3_python.STEREO = 1
 *       orbslam3_python.MONOCULAR = 0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>

#include "System.h"
#include "Atlas.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"

namespace py = pybind11;

/**
 * Thin C++ wrapper around ORB_SLAM3::System that provides numpy-friendly
 * input/output without requiring consumers to deal with OpenCV cv::Mat
 * or Eigen types directly.
 */
class SLAM {
public:
    /**
     * @param vocab      Path to ORBvoc.txt
     * @param settings   Path to camera-settings YAML
     * @param sensor     Sensor enum (0=MONO, 1=STEREO, 2=RGBD, …)
     */
    SLAM(const std::string &vocab,
         const std::string &settings,
         int sensor)
        : sys_(vocab, settings,
               static_cast<ORB_SLAM3::System::eSensor>(sensor),
               /*bUseViewer=*/false),
          last_loop_flag_(false) {}

    /*  tracking*/

    /**
     * Run one RGBD tracking step.
     *
     * @param rgb   uint8 numpy array (H, W, 3)  — BGR or RGB
     * @param depth float32 numpy array (H, W)   — depth in metres
     * @param ts    timestamp (seconds)
     * @return 4×4 float32 camera-to-world SE3 pose (identity on failure)
     */
    py::array_t<float> track_rgbd(py::array_t<uint8_t, py::array::c_style> rgb,
                                   py::array_t<float, py::array::c_style> depth,
                                   double ts) {
        auto r = rgb.request();
        auto d = depth.request();

        if (r.ndim != 3 || d.ndim != 2)
            throw std::runtime_error("rgb must be (H,W,3), depth must be (H,W)");

        int h = static_cast<int>(r.shape[0]);
        int w = static_cast<int>(r.shape[1]);

        cv::Mat rgb_mat(h, w, CV_8UC3, r.ptr);
        cv::Mat dep_mat(static_cast<int>(d.shape[0]),
                        static_cast<int>(d.shape[1]),
                        CV_32FC1, d.ptr);

        Sophus::SE3f pose = sys_.TrackRGBD(rgb_mat, dep_mat, ts);

        // Store trajectory
        Eigen::Matrix4f mat = pose.matrix();
        trajectory_.push_back(mat);

        // Copy into numpy array (row-major)
        auto out = py::array_t<float>(std::vector<py::ssize_t>{4, 4});
        auto buf = out.mutable_unchecked<2>();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                buf(i, j) = mat(i, j);

        return out;
    }

    /*  map queries  */

    /**
     * Returns true once after any loop closure or global BA.
     */
    bool map_changed() {
        return sys_.MapChanged();
    }

    /**
     * Get all 3D map points from the current Atlas.
     *
     * @return (N, 3) float32 numpy array of world-frame XYZ positions.
     */
    py::array_t<float> get_all_map_points() {
        ORB_SLAM3::Atlas *atlas = sys_.mpAtlas;
        if (!atlas) return py::array_t<float>(std::vector<py::ssize_t>{0, 3});

        std::vector<ORB_SLAM3::MapPoint*> points = atlas->GetAllMapPoints();

        // Filter out bad / null points
        std::vector<Eigen::Vector3f> valid;
        valid.reserve(points.size());
        for (auto *mp : points) {
            if (mp && !mp->isBad()) {
                valid.push_back(mp->GetWorldPos());
            }
        }

        size_t n = valid.size();
        auto out = py::array_t<float>(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n),
                                       static_cast<py::ssize_t>(3)});
        auto buf = out.mutable_unchecked<2>();
        for (size_t i = 0; i < n; ++i) {
            buf(i, 0) = valid[i].x();
            buf(i, 1) = valid[i].y();
            buf(i, 2) = valid[i].z();
        }
        return out;
    }

    /**
     * Get poses for all keyframes as (K, 4, 4) float32 numpy array.
     */
    py::array_t<float> get_all_keyframe_poses() {
        ORB_SLAM3::Atlas *atlas = sys_.mpAtlas;
        if (!atlas) return py::array_t<float>(std::vector<py::ssize_t>{0, 4, 4});

        std::vector<ORB_SLAM3::KeyFrame*> kfs = atlas->GetAllKeyFrames();

        // Filter bad keyframes
        std::vector<Eigen::Matrix4f> poses;
        poses.reserve(kfs.size());
        for (auto *kf : kfs) {
            if (kf && !kf->isBad()) {
                Sophus::SE3f se3 = kf->GetPose().inverse();  // cam-to-world
                poses.push_back(se3.matrix());
            }
        }

        size_t k = poses.size();
        auto out = py::array_t<float>(std::vector<py::ssize_t>{static_cast<py::ssize_t>(k),
                                       static_cast<py::ssize_t>(4),
                                       static_cast<py::ssize_t>(4)});
        auto buf = out.mutable_unchecked<3>();
        for (size_t i = 0; i < k; ++i)
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    buf(i, r, c) = poses[i](r, c);

        return out;
    }

    /**
     * Get the full trajectory (one 4×4 pose per TrackRGBD call) as a list.
     */
    py::list get_trajectory() {
        py::list out;
        for (auto &mat : trajectory_) {
            auto arr = py::array_t<float>(std::vector<py::ssize_t>{4, 4});
            auto buf = arr.mutable_unchecked<2>();
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    buf(i, j) = mat(i, j);
            out.append(arr);
        }
        return out;
    }

    /*  loop closure*/

    /**
     * Check if a loop closure has been detected.
     * Uses MapChanged() as a proxy — it fires once per loop closure / global BA.
     */
    bool is_loop_detected() {
        bool changed = sys_.MapChanged();
        if (changed && !last_loop_flag_) {
            last_loop_flag_ = true;
            return true;
        }
        if (!changed) last_loop_flag_ = false;
        return false;
    }

    /*  mode control*/

    void activate_localization_mode() { sys_.ActivateLocalizationMode(); }
    void deactivate_localization_mode() { sys_.DeactivateLocalizationMode(); }
    int get_tracking_state() { return sys_.GetTrackingState(); }

    /*  lifecycle  */

    void shutdown() { sys_.Shutdown(); }

    size_t trajectory_length() const { return trajectory_.size(); }

private:
    ORB_SLAM3::System sys_;
    std::vector<Eigen::Matrix4f> trajectory_;
    bool last_loop_flag_;
};


PYBIND11_MODULE(orbslam3_python, m) {
    m.doc() = "Minimal pybind11 wrapper for ORB_SLAM3 — pilot_core pipeline";

    py::class_<SLAM>(m, "System")
        .def(py::init<const std::string &, const std::string &, int>(),
             py::arg("vocab"), py::arg("settings"), py::arg("sensor_type"))
        .def("track_rgbd", &SLAM::track_rgbd,
             py::arg("rgb"), py::arg("depth"), py::arg("timestamp"),
             "Run one RGBD tracking step, returns 4x4 SE3 pose as float32 numpy array")
        .def("map_changed", &SLAM::map_changed,
             "Returns True once after each loop closure or global BA")
        .def("get_all_map_points", &SLAM::get_all_map_points,
             "Get all 3D map points as (N,3) float32 numpy array")
        .def("get_all_keyframe_poses", &SLAM::get_all_keyframe_poses,
             "Get all keyframe poses as (K,4,4) float32 numpy array")
        .def("get_trajectory", &SLAM::get_trajectory,
             "Get full trajectory as list of 4x4 float32 numpy arrays")
        .def("is_loop_detected", &SLAM::is_loop_detected,
             "Check if a loop closure has been detected since last call")
        .def("activate_localization_mode", &SLAM::activate_localization_mode,
             "Switch to localization-only mode (no mapping)")
        .def("deactivate_localization_mode", &SLAM::deactivate_localization_mode,
             "Re-enable mapping")
        .def("get_tracking_state", &SLAM::get_tracking_state,
             "Return ORB_SLAM3 tracking state enum")
        .def("shutdown", &SLAM::shutdown,
             "Shut down the SLAM system — must call before exit")
        .def("trajectory_length", &SLAM::trajectory_length,
             "Number of poses in the stored trajectory");

    // Sensor-type constants
    m.attr("MONOCULAR")    = static_cast<int>(ORB_SLAM3::System::MONOCULAR);
    m.attr("STEREO")       = static_cast<int>(ORB_SLAM3::System::STEREO);
    m.attr("RGBD")         = static_cast<int>(ORB_SLAM3::System::RGBD);
    m.attr("IMU_MONOCULAR")= static_cast<int>(ORB_SLAM3::System::IMU_MONOCULAR);
    m.attr("IMU_STEREO")   = static_cast<int>(ORB_SLAM3::System::IMU_STEREO);
    m.attr("IMU_RGBD")     = static_cast<int>(ORB_SLAM3::System::IMU_RGBD);
}

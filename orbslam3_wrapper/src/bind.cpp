/*
 * pybind11 wrapper for ORB_SLAM3. Only exposes what the pilot_core workers
 * actually use:
 *
 *   orbslam3_python.System(vocab, settings, sensor_type)
 *       .track_rgbd(rgb, depth, timestamp) -> np.ndarray (4x4 float32, Tcw)
 *       .map_changed()                     -> bool
 *       .map_info()                        -> (map_id, init_kf_id, n_keyframes)
 *       .get_all_map_points()              -> np.ndarray (N x 3 float32)
 *       .get_all_keyframe_poses()          -> np.ndarray (K x 4 x 4 float32, Twc)
 *       .activate_localization_mode()      -> None
 *       .deactivate_localization_mode()    -> None
 *       .get_tracking_state()              -> int
 *       .shutdown()                        -> None
 *
 * plus the ORB_SLAM3::System::eSensor constants MONOCULAR, STEREO, RGBD.
 *
 * Map points and keyframes are those of the Atlas' *current* map. ORB-SLAM3
 * starts a new map (new id) when tracking is lost for good and resets the
 * active map in place (same id, new first keyframe) when it is lost right
 * after initialisation. Either way the world frame changes, so callers should
 * watch map_info(): a change of (map_id, init_kf_id) means "new world frame".
 *
 * Build with orbslam3_wrapper/CMakeLists.txt, docs/slam.md has the details.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <map>
#include <tuple>

#include <opencv2/core.hpp>

#include "System.h"
#include "Atlas.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"

namespace py = pybind11;

// Wraps ORB_SLAM3::System so Python only ever sees numpy arrays, never
// cv::Mat or Eigen types.
class SLAM {
public:
    // vocab = ORBvoc.txt path, settings = camera YAML, sensor = eSensor value
    SLAM(const std::string &vocab,
         const std::string &settings,
         int sensor)
        : sys_(vocab, settings,
               static_cast<ORB_SLAM3::System::eSensor>(sensor),
               /*bUseViewer=*/false) {}

    /**
     * One RGBD tracking step.
     *
     * @param rgb   uint8 (H, W, 3); channel order must match Camera.RGB in the settings
     * @param depth float32 (H, W), metres, same H and W as rgb
     * @param ts    timestamp in seconds (capture time)
     * @return 4x4 float32 Tcw (world-to-camera). When tracking isn't OK this
     *         is identity or stale, so check get_tracking_state().
     */
    py::array_t<float> track_rgbd(py::array_t<uint8_t, py::array::c_style> rgb,
                                   py::array_t<float, py::array::c_style> depth,
                                   double ts) {
        auto r = rgb.request();
        auto d = depth.request();

        if (r.ndim != 3 || r.shape[2] != 3)
            throw std::invalid_argument("rgb must be uint8 (H, W, 3)");
        if (d.ndim != 2 || d.shape[0] != r.shape[0] || d.shape[1] != r.shape[1])
            throw std::invalid_argument("depth must be float32 (H, W) with the same H, W as rgb");

        int h = static_cast<int>(r.shape[0]);
        int w = static_cast<int>(r.shape[1]);

        cv::Mat rgb_mat(h, w, CV_8UC3, r.ptr);
        cv::Mat dep_mat(h, w, CV_32FC1, d.ptr);

        Sophus::SE3f pose = sys_.TrackRGBD(rgb_mat, dep_mat, ts);
        Eigen::Matrix4f mat = pose.matrix();

        // Eigen is column-major by default, so copy element-wise into the row-major array
        auto out = py::array_t<float>(std::vector<py::ssize_t>{4, 4});
        auto buf = out.mutable_unchecked<2>();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                buf(i, j) = mat(i, j);
        return out;
    }

    /*
     * True once after each loop closure / global BA on the current map.
     *
     * System::MapChanged() keeps its last-seen index in a function-static
     * counter shared by all maps, so after a new map is created (whose index
     * starts at 0 again) it stays false until the new map has had more big
     * changes than the old one. Here the last-seen index is kept per map id.
     */
    bool map_changed() {
        ORB_SLAM3::Map *m = sys_.mpAtlas->GetCurrentMap();
        const long unsigned id = m->GetId();
        const int idx = m->GetLastBigChangeIdx();
        auto it = last_big_change_.find(id);
        const int seen = (it == last_big_change_.end()) ? 0 : it->second;
        if (idx > seen) {
            last_big_change_[id] = idx;
            return true;
        }
        return false;
    }

    // (map_id, init_kf_id, n_keyframes) of the Atlas' current map. The pair
    // (map_id, init_kf_id) changes whenever the world frame is redefined.
    std::tuple<long unsigned, long unsigned, long unsigned> map_info() {
        ORB_SLAM3::Map *m = sys_.mpAtlas->GetCurrentMap();
        return std::make_tuple(m->GetId(), m->GetInitKFid(), m->KeyFramesInMap());
    }

    /* Map points of the current map, (N, 3) float32 world-frame XYZ. */
    py::array_t<float> get_all_map_points() {
        std::vector<ORB_SLAM3::MapPoint*> points = sys_.mpAtlas->GetAllMapPoints();

        // skip null and culled (isBad) points
        std::vector<Eigen::Vector3f> valid;
        valid.reserve(points.size());
        for (auto *mp : points) {
            if (mp && !mp->isBad()) valid.push_back(mp->GetWorldPos());
        }

        size_t n = valid.size();
        auto out = py::array_t<float>(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n), 3});
        auto buf = out.mutable_unchecked<2>();
        for (size_t i = 0; i < n; ++i) {
            buf(i, 0) = valid[i].x();
            buf(i, 1) = valid[i].y();
            buf(i, 2) = valid[i].z();
        }
        return out;
    }

    // Keyframe poses of the current map as Twc (camera-to-world, unlike
    // track_rgbd), (K, 4, 4) float32, sorted by keyframe id (= time order).
    py::array_t<float> get_all_keyframe_poses() {
        std::vector<ORB_SLAM3::KeyFrame*> kfs = sys_.mpAtlas->GetAllKeyFrames();
        std::vector<ORB_SLAM3::KeyFrame*> good;
        good.reserve(kfs.size());
        for (auto *kf : kfs)
            if (kf && !kf->isBad()) good.push_back(kf);
        std::sort(good.begin(), good.end(),
                  [](ORB_SLAM3::KeyFrame *a, ORB_SLAM3::KeyFrame *b) { return a->mnId < b->mnId; });

        size_t k = good.size();
        auto out = py::array_t<float>(std::vector<py::ssize_t>{static_cast<py::ssize_t>(k), 4, 4});
        auto buf = out.mutable_unchecked<3>();
        for (size_t i = 0; i < k; ++i) {
            Eigen::Matrix4f T = good[i]->GetPoseInverse().matrix();  // Twc
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    buf(i, r, c) = T(r, c);
        }
        return out;
    }

    void activate_localization_mode() { sys_.ActivateLocalizationMode(); }
    void deactivate_localization_mode() { sys_.DeactivateLocalizationMode(); }
    int get_tracking_state() { return sys_.GetTrackingState(); }

    void shutdown() { sys_.Shutdown(); }

private:
    ORB_SLAM3::System sys_;
    std::map<long unsigned, int> last_big_change_;  // map id -> last reported big-change index
};


PYBIND11_MODULE(orbslam3_python, m) {
    m.doc() = "Minimal pybind11 wrapper for ORB_SLAM3 — pilot_core pipeline";

    py::class_<SLAM>(m, "System")
        .def(py::init<const std::string &, const std::string &, int>(),
             py::arg("vocab"), py::arg("settings"), py::arg("sensor_type"))
        .def("track_rgbd", &SLAM::track_rgbd,
             py::arg("rgb"), py::arg("depth"), py::arg("timestamp"),
             "Run one RGBD tracking step, returns the 4x4 Tcw pose as float32")
        .def("map_changed", &SLAM::map_changed,
             "True once after each loop closure / global BA on the current map")
        .def("map_info", &SLAM::map_info,
             "(map_id, init_kf_id, n_keyframes) of the current map")
        .def("get_all_map_points", &SLAM::get_all_map_points,
             "Current map's points as (N,3) float32")
        .def("get_all_keyframe_poses", &SLAM::get_all_keyframe_poses,
             "Current map's keyframe poses (Twc) as (K,4,4) float32, in id order")
        .def("activate_localization_mode", &SLAM::activate_localization_mode,
             "Switch to localization-only mode (no mapping)")
        .def("deactivate_localization_mode", &SLAM::deactivate_localization_mode,
             "Re-enable mapping")
        .def("get_tracking_state", &SLAM::get_tracking_state,
             "Return ORB_SLAM3 tracking state enum")
        .def("shutdown", &SLAM::shutdown,
             "Shut down the SLAM system — must call before exit");

    m.attr("MONOCULAR") = static_cast<int>(ORB_SLAM3::System::MONOCULAR);
    m.attr("STEREO")    = static_cast<int>(ORB_SLAM3::System::STEREO);
    m.attr("RGBD")      = static_cast<int>(ORB_SLAM3::System::RGBD);
}

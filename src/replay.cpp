// replay.cpp
// Read RGBD frames written by examples/StereoDepth/rgb_depth_aligned.cpp
// Each file format: int width, int height, int channels, int type, raw bytes (width*height*elemSize)

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <thread>
#include <limits>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

static void printUsage(const char* prog) {
	std::cout << "Usage:\n  " << prog << " --folder <path> [--display] [--help]\n\n"
			  << "Options:\n  --folder <path>   Directory containing frame_*.bin files produced by rgb_depth_aligned\n"
			  << "  --help            Show this help message\n";
}

static bool extractTimestamp(const std::string& filename, const std::string& prefix, long long& timestamp) {
	const std::string suffix = ".bin";
	if(filename.rfind(prefix, 0) != 0) {
		return false;
	}
	if(filename.size() <= prefix.size() + suffix.size()) {
		return false;
	}
	std::string tsStr = filename.substr(prefix.size(), filename.size() - prefix.size() - suffix.size());
	try {
		timestamp = std::stoll(tsStr);
		return true;
	} catch(const std::exception&) {
		return false;
	}
}

struct KeyPointRecord {
	float x;
	float y;
	float size;
	float angle;
	float response;
	int32_t octave;
	int32_t class_id;
};

struct MapRow {
	long long timestamp = 0;
	int32_t poseRows = 0;
	int32_t poseCols = 0;
	int32_t poseType = 0;
	std::vector<float> pose;
	std::vector<float> mapPoints;
	std::vector<KeyPointRecord> keypoints;
};

static bool readNextMapRow(std::ifstream& ifs, MapRow& row) {
	if(!ifs.read(reinterpret_cast<char*>(&row.timestamp), sizeof(row.timestamp))) {
		return false;
	}
	if(!ifs.read(reinterpret_cast<char*>(&row.poseRows), sizeof(row.poseRows))) {
		return false;
	}
	if(!ifs.read(reinterpret_cast<char*>(&row.poseCols), sizeof(row.poseCols))) {
		return false;
	}
	if(!ifs.read(reinterpret_cast<char*>(&row.poseType), sizeof(row.poseType))) {
		return false;
	}
	size_t poseCount = static_cast<size_t>(row.poseRows) * static_cast<size_t>(row.poseCols);
	row.pose.resize(poseCount);
	if(poseCount > 0) {
		if(!ifs.read(reinterpret_cast<char*>(row.pose.data()), poseCount * sizeof(float))) {
			return false;
		}
	}

	int32_t mapPointCount = 0;
	if(!ifs.read(reinterpret_cast<char*>(&mapPointCount), sizeof(mapPointCount))) {
		return false;
	}
	row.mapPoints.resize(static_cast<size_t>(mapPointCount) * 3);
	if(mapPointCount > 0) {
		if(!ifs.read(reinterpret_cast<char*>(row.mapPoints.data()), row.mapPoints.size() * sizeof(float))) {
			return false;
		}
	}

	int32_t keypointCount = 0;
	if(!ifs.read(reinterpret_cast<char*>(&keypointCount), sizeof(keypointCount))) {
		return false;
	}
	row.keypoints.resize(static_cast<size_t>(keypointCount));
	if(keypointCount > 0) {
		if(!ifs.read(reinterpret_cast<char*>(row.keypoints.data()), row.keypoints.size() * sizeof(KeyPointRecord))) {
			return false;
		}
	}

	return true;
}

static void printMapRow(const MapRow& row) {
	std::cout << "[map] ts=" << row.timestamp
			  << " pose(" << row.poseRows << "x" << row.poseCols << ", type=" << row.poseType << ")" << std::endl;
	if(!row.pose.empty()) {
		std::cout << "  pose values:";
		for(size_t idx = 0; idx < row.pose.size(); ++idx) {
			if(idx % row.poseCols == 0) {
				std::cout << "\n    ";
			}
			std::cout << row.pose[idx] << ' ';
		}
		std::cout << std::endl;
	}

	size_t mapPointTriples = row.mapPoints.size() / 3;
	std::cout << "  map points delta (" << mapPointTriples << "):";
	for(size_t i = 0; i < row.mapPoints.size(); i += 3) {
		std::cout << " [" << row.mapPoints[i] << ", " << row.mapPoints[i+1] << ", " << row.mapPoints[i+2] << "]";
	}
	std::cout << std::endl;

	std::cout << "  keypoints (" << row.keypoints.size() << "):";
	for(const auto& kp : row.keypoints) {
		std::cout << " (pt=" << kp.x << "," << kp.y
				  << " size=" << kp.size
				  << " angle=" << kp.angle
				  << " resp=" << kp.response
				  << " oct=" << kp.octave
				  << " cls=" << kp.class_id << ")";
	}
	std::cout << std::endl;
}

int main(int argc, char** argv) {
	if(argc < 2) {
		printUsage(argv[0]);
		return 0;
	}

	std::string folder;
	bool display = true;

	for(int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if(a == "--help" || a == "-h") {
			printUsage(argv[0]);
			return 0;
		} else if(a == "--folder" && i + 1 < argc) {
			folder = argv[++i];
		} 
	}

	if(folder.empty()) {
		std::cerr << "Error: --folder <path> is required\n";
		printUsage(argv[0]);
		return 1;
	}

	if(!fs::exists(folder) || !fs::is_directory(folder)) {
		std::cerr << "Error: folder does not exist or is not a directory: " << folder << std::endl;
		return 1;
	}

	// Collect matching files
	std::vector<fs::path> files;
	fs::path mapPointsFile;
	for(auto &ent : fs::directory_iterator(folder)) {
		if(!ent.is_regular_file()) continue;
		auto p = ent.path();
		auto name = p.filename().string();
		// match prefix frame_ and .bin suffix
		if(name.rfind("rgbd_frame_", 0) == 0 && p.extension() == ".bin") {
			files.push_back(p);
		}
		if(name.rfind("map_points_", 0) == 0) {
			if(mapPointsFile.empty() || name < mapPointsFile.filename().string()) {
				mapPointsFile = p;
			}
		}
	}

	if(files.empty()) {
		std::cerr << "No frame_*.bin files found in " << folder << std::endl;
		return 1;
	}

	// Sort by filename (timestamp in the name makes lexicographic order chronological)
	std::sort(files.begin(), files.end(), [](const fs::path &a, const fs::path &b){
		return a.filename().string() < b.filename().string();
	});

	std::cout << "Found " << files.size() << " frames in " << folder << std::endl;

	std::ifstream mapIfs;
	bool hasMapRows = false;
	MapRow currentMapRow;
	if(!mapPointsFile.empty()) {
		mapIfs.open(mapPointsFile, std::ios::binary);
		if(!mapIfs) {
			std::cerr << "Failed to open map points file: " << mapPointsFile << std::endl;
		} else {
			hasMapRows = readNextMapRow(mapIfs, currentMapRow);
			if(!hasMapRows) {
				std::cerr << "Map points file appears empty: " << mapPointsFile << std::endl;
			}
		}
	} else {
		std::cout << "No map_points_* file found alongside frames." << std::endl;
	}

	cv::namedWindow("replay", cv::WINDOW_AUTOSIZE);

	int frameIdx = 0;
	for(auto &p : files) {
		++frameIdx;
		std::ifstream ifs(p, std::ios::binary);
		if(!ifs) {
			std::cerr << "Failed to open " << p << std::endl;
			continue;
		}

		int width=0, height=0, channels=0, type=0;
		ifs.read(reinterpret_cast<char*>(&width), sizeof(width));
		ifs.read(reinterpret_cast<char*>(&height), sizeof(height));
		ifs.read(reinterpret_cast<char*>(&channels), sizeof(channels));
		ifs.read(reinterpret_cast<char*>(&type), sizeof(type));

		if(!ifs) {
			std::cerr << "Failed to read header from " << p << std::endl;
			continue;
		}

		if(width <= 0 || height <= 0 || channels <= 0) {
			std::cerr << "Invalid dimensions in " << p << " (" << width << "x" << height << "x" << channels << ")\n";
			continue;
		}

		// Reconstruct Mat
		cv::Mat mat(height, width, type);
		size_t bytes = mat.total() * mat.elemSize();
		ifs.read(reinterpret_cast<char*>(mat.data), bytes);
		if(!ifs) {
			std::cerr << "Failed to read image data from " << p << std::endl;
			continue;
		}

		std::cout << "Loaded frame " << frameIdx << ": " << p.filename() << " (" << width << "x" << height << ", type=" << type << ")\n";

		long long frameTimestamp = 0;
		if(!extractTimestamp(p.filename().string(), "rgbd_frame_", frameTimestamp)) {
			std::cerr << "Warning: Unable to parse timestamp from filename " << p.filename() << std::endl;
		}
		long long nextFrameTimestamp = std::numeric_limits<long long>::max();
		if(frameIdx < static_cast<int>(files.size())) {
			long long tmpTs = 0;
			if(frameIdx < static_cast<int>(files.size())) {
				const auto& nextPath = files[frameIdx];
				if(extractTimestamp(nextPath.filename().string(), "rgbd_frame_", tmpTs)) {
					nextFrameTimestamp = tmpTs;
				}
			}
		}

		if(hasMapRows) {
			while(hasMapRows && currentMapRow.timestamp >= frameTimestamp && currentMapRow.timestamp < nextFrameTimestamp) {
				printMapRow(currentMapRow);
				hasMapRows = readNextMapRow(mapIfs, currentMapRow);
			}
		}

		if(display) {
			// Expect CV_32FC4: B,G,R,Depth (all float)
			if(mat.type() == CV_32FC4) {
				std::vector<cv::Mat> ch;
				cv::split(mat, ch);
				if(ch.size() >= 4) {
					cv::Mat rgbFloat;
					std::vector<cv::Mat> rgbCh{ch[0], ch[1], ch[2]};
					cv::merge(rgbCh, rgbFloat);
					cv::Mat rgb8;
					rgbFloat.convertTo(rgb8, CV_8UC3);

					// Depth display: normalize and colorize
					cv::Mat depth = ch[3];
					cv::Mat depthNorm, depth8, depthColor;
					cv::normalize(depth, depthNorm, 0, 255, cv::NORM_MINMAX);
					depthNorm.convertTo(depth8, CV_8UC1);
					cv::applyColorMap(depth8, depthColor, cv::COLORMAP_JET);

					// Blend like original example: default weights
					double rgbW = 0.4, depthW = 0.6;
					cv::Mat depthBgr;
					if(depthColor.channels() == 3) depthBgr = depthColor;
					else cv::cvtColor(depthColor, depthBgr, cv::COLOR_GRAY2BGR);

					cv::Mat blended;
					cv::addWeighted(rgb8, rgbW, depthBgr, depthW, 0.0, blended);

					cv::imshow("replay", blended);
				} else {
					cv::imshow("replay", mat);
				}
			} else {
				// Attempt basic display conversions for other types
				cv::Mat disp;
				if(mat.channels() == 1) {
					double minv, maxv; cv::minMaxLoc(mat, &minv, &maxv);
					cv::Mat tmp; mat.convertTo(tmp, CV_8UC1, 255.0/(std::max(1.0, maxv-minv)), -minv);
					cv::applyColorMap(tmp, disp, cv::COLORMAP_JET);
				} else if(mat.channels() == 3) {
					if(mat.depth() != CV_8U) mat.convertTo(disp, CV_8UC3);
					else disp = mat;
				} else {
					// fallback
					cv::Mat tmp; mat.convertTo(tmp, CV_8UC3);
					disp = tmp;
				}
				cv::imshow("replay", disp);
			}

			int key = cv::waitKey(1);
			if(key == 'q' || key == 'Q') {
				std::cout << "Quit requested\n";
				break;
			}
			// small sleep to avoid busy loop
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}
	}

	std::cout << "Done. Processed " << frameIdx << " frames." << std::endl;
	return 0;
}

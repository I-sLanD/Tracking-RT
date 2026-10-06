#pragma once
#include <Eigen/Core>
#include <deque>
#include <vector>

namespace deepsort {
using Measurement = Eigen::Matrix<double, 4, 1>;
using Mean = Eigen::Matrix<double, 8, 1>;
using Covariance = Eigen::Matrix<double, 8, 8>;
struct Box {
  double x, y, width, height;
};
struct Observation {
  Box box;
  double confidence;
  std::vector<float> feature; // L2-normalized person ReID embedding.
};
enum class State { Tentative, Confirmed, Deleted };
struct Track {
  int id = 0, hits = 1, age = 1, time_since_update = 0;
  int detection_index = -1;
  double confidence = 0;
  State state = State::Tentative;
  Mean mean;
  Covariance covariance;
  std::deque<std::vector<float>> gallery;
  Box box() const;
};
struct Config {
  double max_cosine_distance = 0.2, max_iou_distance = 0.7;
  int max_age = 30, n_init = 3, nn_budget = 100;
};
class KalmanFilter {
public:
  void initiate(const Measurement &, Mean &, Covariance &) const;
  void predict(Mean &, Covariance &) const;
  void update(const Measurement &, Mean &, Covariance &) const;
  double gating_distance(const Measurement &, const Mean &,
                         const Covariance &) const;
};
// Thresholded rectangular Hungarian assignment; returned pairs index input
// rows/columns.
std::vector<std::pair<int, int>>
assign(const std::vector<std::vector<double>> &, double threshold);
double iou(const Box &, const Box &);
Measurement xyah(const Box &);
class Tracker {
public:
  explicit Tracker(Config config = {});
  const std::vector<Track> &step(const std::vector<Observation> &detections);
  int created_tracks() const { return next_id_ - 1; }

private:
  Config config_;
  KalmanFilter kf_;
  int next_id_ = 1;
  size_t feature_dimension_ = 0;
  std::vector<Track> tracks_;
  void observe(Track &, const Observation &, int detection_index);
};
} // namespace deepsort

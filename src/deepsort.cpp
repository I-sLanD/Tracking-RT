// Independent C++ implementation of the DeepSORT algorithm. See THIRD_PARTY.md.
#include "deepsort.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace deepsort {
namespace {
constexpr double position_weight = 1.0 / 20, velocity_weight = 1.0 / 160;
constexpr double chi_square_95_4d = 9.4877;
using ProjectCov = Eigen::Matrix<double, 4, 4>;
ProjectCov projected_covariance(const Mean &mean,
                                const Covariance &covariance) {
  Eigen::Vector4d stddev;
  const double height = mean(3); // Preserve the upstream linear state,
                                 // including lost-track predictions.
  stddev << position_weight * height, position_weight * height, 0.1,
      position_weight * height;
  return covariance.topLeftCorner<4, 4>() +
         stddev.array().square().matrix().asDiagonal().toDenseMatrix();
}
double appearance_distance(const Track &track,
                           const std::vector<float> &feature) {
  double best = 2;
  for (const auto &sample : track.gallery) {
    double dot = 0;
    for (size_t i = 0; i < sample.size(); ++i)
      dot += double(sample[i]) * feature[i];
    best = std::min(best, std::clamp(1 - dot, 0.0, 2.0));
  }
  return best;
}
} // namespace
Measurement xyah(const Box &box) {
  Measurement result;
  result << box.x + box.width / 2, box.y + box.height / 2,
      box.width / box.height, box.height;
  return result;
}
Box Track::box() const {
  const double h = mean(3);
  const double w = mean(2) * h;
  return {mean(0) - w / 2, mean(1) - h / 2, w, h};
}
void KalmanFilter::initiate(const Measurement &measurement, Mean &mean,
                            Covariance &covariance) const {
  mean.setZero();
  mean.head<4>() = measurement;
  Mean stddev;
  const double h = measurement(3);
  stddev << 2 * position_weight * h, 2 * position_weight * h, 1e-2,
      2 * position_weight * h, 10 * velocity_weight * h,
      10 * velocity_weight * h, 1e-5, 10 * velocity_weight * h;
  covariance = stddev.array().square().matrix().asDiagonal();
}
void KalmanFilter::predict(Mean &mean, Covariance &covariance) const {
  Covariance motion = Covariance::Identity();
  motion.topRightCorner<4, 4>().setIdentity(); // Baseline dt=1 processed frame.
  const double h = mean(3);
  Mean stddev;
  stddev << position_weight * h, position_weight * h, 1e-2, position_weight * h,
      velocity_weight * h, velocity_weight * h, 1e-5, velocity_weight * h;
  mean = (motion * mean).eval();
  covariance = (motion * covariance * motion.transpose()).eval();
  covariance.diagonal() += stddev.array().square().matrix();
}
void KalmanFilter::update(const Measurement &measurement, Mean &mean,
                          Covariance &covariance) const {
  const auto projected = projected_covariance(mean, covariance);
  const Eigen::LLT<ProjectCov> decomposition(projected);
  if (decomposition.info() != Eigen::Success)
    throw std::runtime_error("Kalman covariance is not positive definite");
  const Eigen::Matrix<double, 8, 4> cross = covariance.leftCols<4>();
  const Eigen::Matrix<double, 8, 4> gain =
      decomposition.solve(cross.transpose()).transpose();
  mean += gain * (measurement - mean.head<4>());
  covariance -= gain * projected * gain.transpose();
  covariance = ((covariance + covariance.transpose()) * 0.5).eval();
}
double KalmanFilter::gating_distance(const Measurement &measurement,
                                     const Mean &mean,
                                     const Covariance &covariance) const {
  const Eigen::LLT<ProjectCov> decomposition(
      projected_covariance(mean, covariance));
  if (decomposition.info() != Eigen::Success)
    throw std::runtime_error("Invalid gating covariance");
  const Measurement delta =
      decomposition.matrixL().solve(measurement - mean.head<4>());
  return delta.squaredNorm();
}
double iou(const Box &a, const Box &b) {
  const double w = std::max(0.0, std::min(a.x + a.width, b.x + b.width) -
                                     std::max(a.x, b.x));
  const double h = std::max(0.0, std::min(a.y + a.height, b.y + b.height) -
                                     std::max(a.y, b.y));
  const double intersection = w * h;
  return intersection / std::max(1e-12, a.width * a.height +
                                            b.width * b.height - intersection);
}
std::vector<std::pair<int, int>>
assign(const std::vector<std::vector<double>> &cost, double threshold) {
  if (cost.empty() || cost.front().empty())
    return {};
  const int n = int(cost.size()), real_columns = int(cost.front().size()),
            m = std::max(n, real_columns);
  for (const auto &row : cost)
    if (int(row.size()) != real_columns)
      throw std::runtime_error("Ragged assignment matrix");
  const double capped = threshold + 1e-5;
  const auto value = [&](int i, int j) {
    return j < real_columns && std::isfinite(cost[i][j])
               ? std::min(cost[i][j], capped)
               : capped;
  };
  std::vector<double> u(n + 1), v(m + 1);
  std::vector<int> p(m + 1), way(m + 1);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minimum(m + 1, std::numeric_limits<double>::infinity());
    std::vector<bool> used(m + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = std::numeric_limits<double>::infinity();
      int j1 = 0;
      for (int j = 1; j <= m; ++j)
        if (!used[j]) {
          const double cur = value(i0 - 1, j - 1) - u[i0] - v[j];
          if (cur < minimum[j]) {
            minimum[j] = cur;
            way[j] = j0;
          }
          if (minimum[j] < delta) {
            delta = minimum[j];
            j1 = j;
          }
        }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else
          minimum[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }
  std::vector<std::pair<int, int>> result;
  for (int j = 1; j <= real_columns; ++j)
    if (p[j] && cost[p[j] - 1][j - 1] <= threshold &&
        std::isfinite(cost[p[j] - 1][j - 1]))
      result.emplace_back(p[j] - 1, j - 1);
  std::sort(result.begin(), result.end());
  return result;
}
Tracker::Tracker(Config config) : config_(config) {
  if (!std::isfinite(config.max_cosine_distance) ||
      config.max_cosine_distance < 0 || config.max_cosine_distance > 2 ||
      !std::isfinite(config.max_iou_distance) || config.max_iou_distance < 0 ||
      config.max_iou_distance > 1 || config.max_age < 1 || config.n_init < 1 ||
      config.nn_budget < 1)
    throw std::runtime_error("Invalid DeepSORT configuration");
}
void Tracker::observe(Track &track, const Observation &detection, int index) {
  kf_.update(xyah(detection.box), track.mean, track.covariance);
  ++track.hits;
  track.time_since_update = 0;
  track.detection_index = index;
  track.confidence = detection.confidence;
  track.gallery.push_back(detection.feature);
  while (int(track.gallery.size()) > config_.nn_budget)
    track.gallery.pop_front();
  if (track.hits >= config_.n_init)
    track.state = State::Confirmed;
}
const std::vector<Track> &
Tracker::step(const std::vector<Observation> &detections) {
  // Validate all input before mutating track state.
  size_t dimension = feature_dimension_;
  for (const auto &d : detections) {
    if (!std::isfinite(d.box.x) || !std::isfinite(d.box.y) ||
        !std::isfinite(d.box.width) || !std::isfinite(d.box.height) ||
        d.box.width <= 0 || d.box.height <= 0 || !std::isfinite(d.confidence) ||
        d.confidence < 0 || d.confidence > 1 || d.feature.empty())
      throw std::runtime_error("Invalid observation");
    if (!dimension)
      dimension = d.feature.size();
    if (d.feature.size() != dimension)
      throw std::runtime_error("ReID feature dimension changed");
    double norm = 0;
    for (float v : d.feature) {
      if (!std::isfinite(v))
        throw std::runtime_error("Non-finite ReID feature");
      norm += double(v) * v;
    }
    if (std::abs(norm - 1) > 1e-3)
      throw std::runtime_error("Expected unit-normalized ReID feature");
  }
  feature_dimension_ = dimension;
  for (auto &track : tracks_) {
    kf_.predict(track.mean, track.covariance);
    ++track.age;
    ++track.time_since_update;
    track.detection_index = -1;
  }
  std::vector<bool> matched_track(tracks_.size()),
      matched_detection(detections.size());
  const auto match = [&](const std::vector<int> &rows, bool appearance) {
    std::vector<int> columns;
    for (size_t j = 0; j < detections.size(); ++j)
      if (!matched_detection[j])
        columns.push_back(int(j));
    std::vector<std::vector<double>> cost(rows.size(),
                                          std::vector<double>(columns.size()));
    for (size_t i = 0; i < rows.size(); ++i)
      for (size_t j = 0; j < columns.size(); ++j) {
        const auto &track = tracks_[rows[i]];
        const auto &detection = detections[columns[j]];
        if (appearance) {
          cost[i][j] = kf_.gating_distance(xyah(detection.box), track.mean,
                                           track.covariance) <= chi_square_95_4d
                           ? appearance_distance(track, detection.feature)
                           : 1e5;
        } else
          cost[i][j] = 1 - iou(track.box(), detection.box);
      }
    for (const auto &pair :
         assign(cost, appearance ? config_.max_cosine_distance
                                 : config_.max_iou_distance)) {
      const int i = rows[pair.first], j = columns[pair.second];
      matched_track[i] = true;
      matched_detection[j] = true;
      observe(tracks_[i], detections[j], j);
    }
  };
  // Matching cascade prioritizes recently observed confirmed tracks.
  for (int age = 1; age <= config_.max_age; ++age) {
    std::vector<int> rows;
    for (size_t i = 0; i < tracks_.size(); ++i)
      if (!matched_track[i] && tracks_[i].state == State::Confirmed &&
          tracks_[i].time_since_update == age)
        rows.push_back(int(i));
    if (!rows.empty())
      match(rows, true);
  }
  std::vector<int> iou_rows;
  for (size_t i = 0; i < tracks_.size(); ++i)
    if (!matched_track[i] && (tracks_[i].state == State::Tentative ||
                              tracks_[i].time_since_update == 1))
      iou_rows.push_back(int(i));
  match(iou_rows, false);
  for (size_t i = 0; i < tracks_.size(); ++i)
    if (!matched_track[i]) {
      if (tracks_[i].state == State::Tentative ||
          tracks_[i].time_since_update > config_.max_age)
        tracks_[i].state = State::Deleted;
    }
  tracks_.erase(
      std::remove_if(tracks_.begin(), tracks_.end(),
                     [](const Track &t) { return t.state == State::Deleted; }),
      tracks_.end());
  for (size_t j = 0; j < detections.size(); ++j)
    if (!matched_detection[j]) {
      Track track;
      track.id = next_id_++;
      track.detection_index = int(j);
      track.confidence = detections[j].confidence;
      kf_.initiate(xyah(detections[j].box), track.mean, track.covariance);
      track.gallery.push_back(detections[j].feature);
      if (config_.n_init == 1)
        track.state = State::Confirmed;
      tracks_.push_back(std::move(track));
    }
  return tracks_;
}
} // namespace deepsort

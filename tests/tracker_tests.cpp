#include "deepsort.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace deepsort;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
Observation person(double x, float a = 1, float b = 0) {
  return {{x, 50, 40, 100}, .9, {a, b}};
}
const Track &observed(const std::vector<Track> &tracks, int detection) {
  for (const auto &t : tracks)
    if (t.detection_index == detection)
      return t;
  throw std::runtime_error("Expected observation has no track");
}
void tests() {
  // Compare rectangular assignment against exhaustive optimal matching,
  // independent of solver implementation.
  std::mt19937 random(17);
  std::uniform_real_distribution<double> uniform(.01, .99);
  for (int n = 1; n <= 4; ++n)
    for (int m = n; m <= 5; ++m)
      for (int trial = 0; trial < 15; ++trial) {
        std::vector<std::vector<double>> costs(n, std::vector<double>(m));
        for (auto &row : costs)
          for (auto &v : row)
            v = uniform(random);
        std::vector<int> permutation(m);
        for (int j = 0; j < m; ++j)
          permutation[j] = j;
        double best = 1e10;
        do {
          double value = 0;
          for (int i = 0; i < n; ++i)
            value += costs[i][permutation[i]];
          best = std::min(best, value);
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        auto matches = assign(costs, 1);
        double actual = 0;
        for (auto p : matches)
          actual += costs[p.first][p.second];
        require(matches.size() == size_t(n) && std::abs(actual - best) < 1e-10,
                "Hungarian assignment is not globally optimal");
      }
  require(assign({{5, 5}, {5, 5}}, .2).empty(), "Forbidden matches accepted");
  require(assign({{.4}, {.1}, {.3}}, .2) ==
              std::vector<std::pair<int, int>>{{1, 0}},
          "Tall rectangular assignment failed");
  std::cout << "PASS optimal rectangular assignment and rejection\n";
  Tracker tentative;
  require(tentative.step({person(50)})[0].state == State::Tentative,
          "Premature confirmation");
  require(tentative.step({}).empty(),
          "Missed tentative track should be deleted");
  Tracker tracker;
  tracker.step({person(50)});
  tracker.step({person(52)});
  require(tracker.step({person(54)})[0].state == State::Confirmed,
          "Track not confirmed after n_init");
  tracker.step({});
  tracker.step({});
  const auto &recovered = tracker.step({person(60)});
  require(observed(recovered, 0).id == 1 && recovered.size() == 1,
          "Short gap changed identity");
  std::cout << "PASS confirmation, tentative deletion and short-gap recovery\n";
  Tracker aging(Config{.2, .7, 3, 3, 2});
  for (int i = 0; i < 6; ++i)
    aging.step({person(50)});
  require(aging.step({})[0].gallery.size() == 2,
          "Feature gallery is unbounded");
  aging.step({});
  aging.step({});
  require(aging.step({}).empty(), "Track survived beyond max_age");
  require(aging.step({person(50)})[0].id == 2, "Deleted ID was reused");
  std::cout << "PASS age boundary, monotonic IDs and bounded gallery\n";
  Tracker crossing;
  for (int i = 0; i < 3; ++i)
    crossing.step({person(100, 1, 0), person(104, 0, 1)});
  const auto &ambiguous = crossing.step({person(101, 0, 1), person(103, 1, 0)});
  require(observed(ambiguous, 0).id == 2 && observed(ambiguous, 1).id == 1,
          "Appearance failed in spatially ambiguous crossing");
  std::cout
      << "PASS appearance-based association with reversed detection order\n";
  Tracker gated;
  for (int i = 0; i < 3; ++i)
    gated.step({person(100)});
  const auto &far = gated.step({person(1000)});
  require(observed(far, 0).id == 2, "Motion gate accepted implausible jump");
  std::cout << "PASS Mahalanobis motion gating\n";
  Tracker invalid;
  invalid.step({person(50)});
  bool rejected = false;
  try {
    auto d = person(50);
    d.feature = {0, 0};
    invalid.step({d});
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected, "Zero embedding accepted");
  require(invalid.step({person(51)})[0].age == 2,
          "Invalid input mutated state");
  KalmanFilter kf;
  Mean mean;
  Covariance covariance;
  kf.initiate(xyah(person(50).box), mean, covariance);
  for (int i = 0; i < 1000; ++i) {
    kf.predict(mean, covariance);
    kf.update(xyah(person(50 + i * .2).box), mean, covariance);
  }
  require(mean.allFinite() && covariance.allFinite() &&
              (covariance.diagonal().array() > 0).all(),
          "Kalman state became invalid");
  require((covariance - covariance.transpose()).cwiseAbs().maxCoeff() < 1e-10,
          "Kalman covariance lost symmetry");
  std::cout << "PASS input rejection and long-run Kalman stability\n";
}
void replay(const char *input, const char *output) {
  std::ifstream f(input);
  std::ofstream g(output);
  int frames = 0, dimension = 0;
  f >> frames >> dimension;
  if (!f || !g || frames < 1 || frames > 100000 || dimension < 1 ||
      dimension > 4096)
    throw std::runtime_error("Invalid replay header");
  Tracker tracker;
  g << std::setprecision(17)
    << "frame,id,hits,age,time_since_update,state,detection_index,x,y,w,h";
  for (int i = 0; i < 8; ++i)
    g << ",mean" << i;
  for (int i = 0; i < 64; ++i)
    g << ",cov" << i;
  g << "\n";
  for (int frame = 1; frame <= frames; ++frame) {
    int count = -1;
    f >> count;
    if (!f || count < 0 || count > 10000)
      throw std::runtime_error("Invalid replay count");
    std::vector<Observation> detections(count);
    for (auto &d : detections) {
      f >> d.box.x >> d.box.y >> d.box.width >> d.box.height >> d.confidence;
      d.feature.resize(dimension);
      for (auto &v : d.feature)
        f >> v;
    }
    if (!f)
      throw std::runtime_error("Truncated replay input");
    for (const auto &t : tracker.step(detections)) {
      const auto b = t.box();
      g << frame << "," << t.id << "," << t.hits << "," << t.age << ","
        << t.time_since_update << "," << int(t.state) << ","
        << t.detection_index << "," << b.x << "," << b.y << "," << b.width
        << "," << b.height;
      for (int i = 0; i < 8; ++i)
        g << "," << t.mean(i);
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          g << "," << t.covariance(i, j);
      g << "\n";
    }
  }
  if (!g)
    throw std::runtime_error("Replay output failed");
}
int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--replay")
      replay(argv[2], argv[3]);
    else if (argc == 1)
      tests();
    else
      throw std::runtime_error("Usage: tracker_tests [--replay input output]");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}

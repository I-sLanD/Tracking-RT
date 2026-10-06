#include "deepsort.hpp"
#include "inference.hpp"

struct TrackingOptions {
  fs::path model, reid, runtime, frames, output;
  std::string provider = "dml";
  int threads = 4, device = 0, limit = 0, warmup = 3;
  double fps = 30;
  float threshold = 0.25f;
  bool dump_features = false, profile = false;
  deepsort::Config config;
};
int track(const std::vector<std::string> &args) {
  TrackingOptions o;
  for (size_t i = 1; i < args.size(); ++i) {
    const auto &key = args[i];
    if (key == "--help") {
      std::cout << "yolo26_track --model yolo.onnx --reid osnet.onnx --runtime "
                   "runtime.dll --frames PNG_directory --output directory "
                   "[--provider cpu|dml] [--fps 30] [--limit 0] [--threshold "
                   ".25] [--max-age 30] [--n-init 3] "
                   "[--max-cosine .2] [--max-iou .7] [--nn-budget 100] "
                   "[--dump-features] [--profile]\n";
      return 0;
    }
    if (key == "--dump-features") {
      o.dump_features = true;
      continue;
    }
    if (key == "--profile") {
      o.profile = true;
      continue;
    }
    if (i + 1 >= args.size())
      throw std::runtime_error("Missing value for " + key);
    const auto value = args[++i];
    if (key == "--model")
      o.model = fs::u8path(value);
    else if (key == "--reid")
      o.reid = fs::u8path(value);
    else if (key == "--runtime")
      o.runtime = fs::u8path(value);
    else if (key == "--frames")
      o.frames = fs::u8path(value);
    else if (key == "--output")
      o.output = fs::u8path(value);
    else if (key == "--provider")
      o.provider = value;
    else if (key == "--threads")
      o.threads = std::stoi(value);
    else if (key == "--device")
      o.device = std::stoi(value);
    else if (key == "--limit")
      o.limit = std::stoi(value);
    else if (key == "--warmup")
      o.warmup = std::stoi(value);
    else if (key == "--fps")
      o.fps = std::stod(value);
    else if (key == "--threshold")
      o.threshold = std::stof(value);
    else if (key == "--max-age")
      o.config.max_age = std::stoi(value);
    else if (key == "--n-init")
      o.config.n_init = std::stoi(value);
    else if (key == "--max-cosine")
      o.config.max_cosine_distance = std::stod(value);
    else if (key == "--max-iou")
      o.config.max_iou_distance = std::stod(value);
    else if (key == "--nn-budget")
      o.config.nn_budget = std::stoi(value);
    else
      throw std::runtime_error("Unknown option " + key);
  }
  if (o.model.empty() || o.reid.empty() || o.runtime.empty() ||
      o.frames.empty() || o.output.empty())
    throw std::runtime_error(
        "Required: --model --reid --runtime --frames --output");
  if (o.threads < 1 || o.device < 0 || o.limit < 0 || o.warmup < 0 ||
      !std::isfinite(o.fps) || o.fps <= 0 || !std::isfinite(o.threshold) ||
      o.threshold < 0 || o.threshold > 1)
    throw std::runtime_error("Invalid numeric option");
  deepsort::Tracker tracker(o.config);
  std::vector<fs::path> frames;
  for (const auto &entry : fs::directory_iterator(o.frames)) {
    if (!entry.is_regular_file())
      continue;
    auto extension = entry.path().extension().u8string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (extension == ".png")
      frames.push_back(entry.path());
  }
  std::sort(frames.begin(),
            frames.end()); // Zero-padded sequence names from video_frames.py.
  if (o.limit && size_t(o.limit) < frames.size())
    frames.resize(o.limit);
  if (frames.empty())
    throw std::runtime_error("No PNG frames; prepare a video sequence first");
  fs::create_directories(o.output);
  Runtime runtime(o.runtime);
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo26-deepsort");
  const auto initialization_start = Clock::now();
  NativeModel detector(env, runtime, o.model, o.provider, o.threads, o.device,
                       o.profile ? o.output / "detector-profile" : fs::path{});
  NativeModel reid(env, runtime, o.reid, o.provider, o.threads, o.device,
                   o.profile ? o.output / "reid-profile" : fs::path{});
  const double initialization = ms(initialization_start, Clock::now());
  const auto &shape = detector.input_shape;
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 ||
      shape[2] != shape[3] || shape[2] > 4096 ||
      detector.output_shape.size() != 3 || detector.output_shape[0] != 1 ||
      detector.output_shape[2] != 6)
    throw std::runtime_error("Expected YOLO26 [1,3,H,H] -> [1,N,6]");
  if (reid.input_shape != std::vector<int64_t>{1, 3, 256, 128} ||
      reid.output_shape != std::vector<int64_t>{1, 512})
    throw std::runtime_error("Expected OSNet [1,3,256,128] -> [1,512]");
  Image first = load_image(frames.front());
  letterbox(first, int(shape[2]), detector.input);
  std::fill(reid.input.begin(), reid.input.end(), 0);
  for (int i = 0; i < o.warmup; ++i) {
    detector.run();
    reid.run();
  }
  std::ofstream jsonl(o.output / "tracks.jsonl"),
      timings(o.output / "timings.csv"), mot(o.output / "mot.txt");
  std::ofstream features;
  if (o.dump_features)
    features.open(o.output / "features.f32", std::ios::binary);
  if (!jsonl || !timings || !mot || (o.dump_features && !features))
    throw std::runtime_error("Cannot open tracking outputs");
  jsonl << std::setprecision(10);
  timings << std::setprecision(10);
  mot << std::setprecision(10);
  timings
      << "frame,people,decode_ms,detector_pre_ms,detector_infer_ms,detector_"
         "post_ms,reid_pre_ms,reid_infer_ms,association_ms,pipeline_ms\n";
  std::vector<std::vector<double>> stage_times(7);
  std::vector<double> pipelines, decoded_times;
  size_t feature_offset = 0, detection_count = 0, confirmed_observations = 0,
         peak_people = 0;
  bool saved_reid_probe = false;
  for (size_t index = 0; index < frames.size(); ++index) {
    const auto d0 = Clock::now();
    Image image = index == 0 ? std::move(first) : load_image(frames[index]);
    const auto t0 = Clock::now();
    if (image.width <= 0 || image.height <= 0)
      throw std::runtime_error("Invalid frame");
    const auto transform = letterbox(image, int(shape[2]), detector.input);
    const auto t1 = Clock::now();
    detector.run();
    const auto t2 = Clock::now();
    const auto all =
        decode(detector.output.data(), size_t(detector.output_shape[1]),
               o.threshold, transform, image, 80);
    std::vector<Detection> people;
    for (const auto &detection : all)
      if (detection.class_id == 0)
        people.push_back(detection);
    const auto t3 = Clock::now();
    std::vector<deepsort::Observation> observations;
    std::vector<std::array<int, 4>> crops;
    std::vector<float> probe_input, probe_output;
    double reid_pre = 0, reid_infer = 0;
    for (const auto &detection : people) {
      const auto r0 = Clock::now();
      crops.push_back(reid_crop(image, detection, reid.input));
      const auto r1 = Clock::now();
      reid.run();
      const auto r2 = Clock::now();
      auto embedding = normalized_embedding(reid.output);
      const auto r3 = Clock::now();
      // Normalize is included in ReID preprocessing/postprocessing total.
      reid_pre += ms(r0, r1) + ms(r2, r3);
      reid_infer += ms(r1, r2);
      observations.push_back(
          {{detection.x1, detection.y1, detection.x2 - detection.x1,
            detection.y2 - detection.y1},
           detection.confidence,
           std::move(embedding)});
      if (o.dump_features && !saved_reid_probe && probe_input.empty()) {
        probe_input = reid.input;
        probe_output = reid.output;
      }
    }
    const auto t4 = Clock::now();
    const auto &tracks = tracker.step(observations);
    const auto t5 = Clock::now();
    const std::array<double, 7> stages{ms(t0, t1), ms(t1, t2), ms(t2, t3),
                                       reid_pre,   reid_infer, ms(t4, t5),
                                       ms(t0, t5)};
    for (size_t s = 0; s < stage_times.size(); ++s)
      stage_times[s].push_back(stages[s]);
    pipelines.push_back(ms(t0, t5));
    decoded_times.push_back(ms(d0, t0));
    detection_count += people.size();
    peak_people = std::max(peak_people, people.size());
    timings << index + 1 << "," << people.size() << "," << ms(d0, t0);
    for (double value : stages)
      timings << "," << value;
    timings << "\n";
    // Serialization/dumps happen after the measured perception pipeline.
    if (!probe_input.empty()) {
      write_floats(o.output / "reid-input.f32", probe_input.data(),
                   probe_input.size());
      write_floats(o.output / "reid-output.f32", probe_output.data(),
                   probe_output.size());
      std::ofstream probe(o.output / "reid-probe.json");
      probe << "{\"file\":\"" << escape(frames[index].filename().u8string())
            << "\",\"crop\":[" << crops[0][0] << "," << crops[0][1] << ","
            << crops[0][2] << "," << crops[0][3] << "]}";
      saved_reid_probe = true;
    }
    jsonl << "{\"frame\":" << index + 1 << ",\"timestamp_s\":" << index / o.fps
          << ",\"file\":\"" << escape(frames[index].filename().u8string())
          << "\",\"detections\":[";
    for (size_t j = 0; j < observations.size(); ++j) {
      if (j)
        jsonl << ",";
      const auto &d = observations[j];
      jsonl << "{\"tlwh\":[" << d.box.x << "," << d.box.y << "," << d.box.width
            << "," << d.box.height << "],\"confidence\":" << d.confidence;
      if (o.dump_features) {
        jsonl << ",\"feature_offset\":" << feature_offset;
        features.write(reinterpret_cast<const char *>(d.feature.data()),
                       std::streamsize(d.feature.size() * sizeof(float)));
        feature_offset += d.feature.size();
      }
      jsonl << "}";
    }
    jsonl << "],\"tracks\":[";
    for (size_t j = 0; j < tracks.size(); ++j) {
      if (j)
        jsonl << ",";
      const auto &track = tracks[j];
      const auto box = track.box();
      const bool observed = track.time_since_update == 0;
      jsonl << "{\"id\":" << track.id << ",\"state\":\""
            << (track.state == deepsort::State::Confirmed ? "confirmed"
                                                          : "tentative")
            << "\",\"observed\":" << (observed ? "true" : "false")
            << ",\"age\":" << track.age << ",\"hits\":" << track.hits
            << ",\"time_since_update\":" << track.time_since_update
            << ",\"detection_index\":" << track.detection_index
            << ",\"confidence\":" << track.confidence
            << ",\"gallery_size\":" << track.gallery.size() << ",\"valid_box\":"
            << ((box.width > 0 && box.height > 0) ? "true" : "false")
            << ",\"tlwh\":[" << box.x << "," << box.y << "," << box.width << ","
            << box.height << "]}";
      if (track.state == deepsort::State::Confirmed && observed &&
          box.width > 0 && box.height > 0) {
        ++confirmed_observations;
        mot << index + 1 << "," << track.id << "," << box.x << "," << box.y
            << "," << box.width << "," << box.height << "," << track.confidence
            << ",-1,-1,-1\n";
      }
    }
    jsonl << "]}\n";
    if ((index + 1) % 60 == 0 || index + 1 == frames.size())
      std::cout << "Frame " << index + 1 << "/" << frames.size()
                << " people=" << people.size()
                << " active_tracks=" << tracks.size() << "\n";
  }
  if (!jsonl || !timings || !mot || (o.dump_features && !features))
    throw std::runtime_error("Tracking output write failed");
  std::string detector_profile, reid_profile;
  if (o.profile) {
    detector_profile = detector.end_profile();
    reid_profile = reid.end_profile();
  }
  std::ofstream summary(o.output / "summary.json");
  if (!summary)
    throw std::runtime_error("Cannot write summary");
  summary
      << std::setprecision(10) << "{\"provider\":\"" << o.provider
      << "\",\"runtime_version\":\"" << runtime.version
      << "\",\"frames\":" << frames.size() << ",\"fps\":" << o.fps
      << ",\"detections\":" << detection_count
      << ",\"peak_people\":" << peak_people
      << ",\"created_tracks\":" << tracker.created_tracks()
      << ",\"confirmed_observations\":" << confirmed_observations
      << ",\"initialization_ms\":" << initialization
      << ",\"config\":{\"max_age\":" << o.config.max_age
      << ",\"n_init\":" << o.config.n_init
      << ",\"max_cosine\":" << o.config.max_cosine_distance
      << ",\"max_iou\":" << o.config.max_iou_distance
      << ",\"nn_budget\":" << o.config.nn_budget
      << ",\"threshold\":" << o.threshold << "},"
      << "\"timing_scope\":\"in-memory YOLO preprocessing/inference/decoding + "
         "per-person ReID preprocessing/inference/normalization + association; "
         "excludes PNG decoding, logging, video IO, model initialization and "
         "warmup\","
      << "\"motion_time_step\":\"one processed frame; max_age is frames, "
         "timestamps are CFR index/fps\",\"reid_batch\":1,\"person_class\":0,"
      << "\"detector_profile\":\"" << escape(detector_profile)
      << "\",\"reid_profile\":\"" << escape(reid_profile)
      << "\",\"timings_ms\":{";
  constexpr std::array<const char *, 7> names{
      "detector_pre", "detector_infer", "detector_post", "reid_pre",
      "reid_infer",   "association",    "pipeline"};
  for (size_t s = 0; s < names.size(); ++s) {
    if (s)
      summary << ",";
    summary << "\"" << names[s] << "\":";
    json_stats(summary, summarize(stage_times[s]));
  }
  summary << ",\"png_decode\":";
  json_stats(summary, summarize(decoded_times));
  summary << "}}\n";
  if (!summary)
    throw std::runtime_error("Summary write failed");
  const auto pipeline = summarize(pipelines);
  std::cout << "DeepSORT provider=" << o.provider
            << " created_tracks=" << tracker.created_tracks()
            << " pipeline_mean_ms=" << pipeline.mean
            << " pipeline_p95_ms=" << pipeline.p95 << "\n";
  return 0;
}
#ifdef _WIN32
int wmain(int argc, wchar_t **argv) {
  std::vector<std::string> args;
  for (int i = 0; i < argc; ++i)
    args.push_back(fs::path(argv[i]).u8string());
#else
int main(int argc, char **argv) {
  std::vector<std::string> args(argv, argv + argc);
#endif
  try {
    return track(args);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

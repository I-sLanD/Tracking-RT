#include "native.hpp"

struct Options {
  fs::path model, image, runtime, labels, output;
  std::string provider = "cpu";
  int iterations = 30, warmup = 3, threads = 4, device = 0;
  float threshold = 0.25f;
  bool profile = false;
};
int run(const std::vector<std::string> &args) {
  Options o;
  for (size_t i = 1; i < args.size(); ++i) {
    const auto &key = args[i];
    if (key == "--help") {
      std::cout << "yolo26_verify --model model.onnx --image image.jpg "
                   "--runtime runtime.dll --labels classes.txt --output "
                   "directory [--provider cpu|dml] [--iterations 30] [--warmup "
                   "3] [--threads 4] [--threshold 0.25] [--profile]\n";
      return 0;
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
    else if (key == "--image")
      o.image = fs::u8path(value);
    else if (key == "--runtime")
      o.runtime = fs::u8path(value);
    else if (key == "--labels")
      o.labels = fs::u8path(value);
    else if (key == "--output")
      o.output = fs::u8path(value);
    else if (key == "--provider")
      o.provider = value;
    else if (key == "--iterations")
      o.iterations = std::stoi(value);
    else if (key == "--warmup")
      o.warmup = std::stoi(value);
    else if (key == "--threads")
      o.threads = std::stoi(value);
    else if (key == "--device")
      o.device = std::stoi(value);
    else if (key == "--threshold")
      o.threshold = std::stof(value);
    else
      throw std::runtime_error("Unknown option " + key);
  }
  if (o.model.empty() || o.image.empty() || o.runtime.empty() ||
      o.labels.empty() || o.output.empty())
    throw std::runtime_error(
        "Required: --model --image --runtime --labels --output");
  if (o.iterations < 1 || o.warmup < 0 || o.threads < 1 || o.device < 0 ||
      !std::isfinite(o.threshold) || o.threshold < 0 || o.threshold > 1)
    throw std::runtime_error("Invalid numeric option");
  if (o.provider != "cpu" && o.provider != "dml")
    throw std::runtime_error("Provider must be cpu or dml");
  fs::create_directories(o.output);
  std::vector<std::string> labels;
  std::ifstream lf(o.labels);
  std::string line;
  while (std::getline(lf, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    labels.push_back(line);
  }
  if (labels.empty())
    throw std::runtime_error("Empty label file");
  Image image = load_image(o.image);
  Runtime runtime(o.runtime); // Outlives all Ort wrappers below.
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo26-cpp");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(o.threads);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  if (o.provider == "dml") {
#ifdef _WIN32
    using AppendDml = OrtStatus *(ORT_API_CALL *)(OrtSessionOptions *, int);
    const auto append = reinterpret_cast<AppendDml>(
        runtime.symbol("OrtSessionOptionsAppendExecutionProvider_DML"));
    if (!append)
      throw std::runtime_error("Runtime does not expose the DirectML provider");
    options.DisableMemPattern();
    Ort::ThrowOnError(append(options, o.device));
#else
    throw std::runtime_error(
        "DirectML is Windows-only; use cpu on this platform");
#endif
  }
  if (o.profile)
    options.EnableProfiling((o.output / "ort-profile").c_str());
  const auto start = Clock::now();
  Ort::Session session(env, fs::absolute(o.model).c_str(), options);
  const double initialization_ms = ms(start, Clock::now());
  if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
    throw std::runtime_error("Expected one input and one output");
  const auto input_shape =
      session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const auto output_shape =
      session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3 ||
      input_shape[2] <= 0 || input_shape[2] != input_shape[3])
    throw std::runtime_error("Expected static float32 input [1,3,H,H]");
  if (output_shape.size() != 3 || output_shape[0] != 1 ||
      output_shape[1] <= 0 || output_shape[2] != 6)
    throw std::runtime_error(
        "Expected end-to-end output [1,N,6]; export explicitly with nms=False");
  if (session.GetInputTypeInfo(0)
              .GetTensorTypeAndShapeInfo()
              .GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      session.GetOutputTypeInfo(0)
              .GetTensorTypeAndShapeInfo()
              .GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("This baseline expects float32 input and output");
  const int side = int(input_shape[2]);
  if (side > 4096)
    throw std::runtime_error("Input side exceeds baseline memory limit");
  std::vector<float> input(size_t(3) * side * side);
  Transform transform = letterbox(image, side, input);
  auto memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_value =
      Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                      input_shape.data(), input_shape.size());
  Ort::AllocatorWithDefaultOptions allocator;
  const auto input_name = session.GetInputNameAllocated(0, allocator);
  const auto output_name = session.GetOutputNameAllocated(0, allocator);
  const char *input_names[] = {input_name.get()};
  const char *output_names[] = {output_name.get()};
  std::vector<Ort::Value> outputs;
  for (int i = 0; i < o.warmup; ++i)
    outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_value,
                          1, output_names, 1);
  std::vector<double> preprocess_times, inference_times, postprocess_times,
      total_times;
  std::vector<Detection> detections;
  for (int i = 0; i < o.iterations; ++i) {
    const auto t0 = Clock::now();
    transform = letterbox(image, side, input);
    const auto t1 = Clock::now();
    outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_value,
                          1, output_names, 1);
    const auto t2 = Clock::now();
    if (outputs[0].GetTensorTypeAndShapeInfo().GetShape() != output_shape)
      throw std::runtime_error("Unexpected runtime output shape");
    detections =
        decode(outputs[0].GetTensorData<float>(), size_t(output_shape[1]),
               o.threshold, transform, image, labels.size());
    const auto t3 = Clock::now();
    preprocess_times.push_back(ms(t0, t1));
    inference_times.push_back(ms(t1, t2));
    postprocess_times.push_back(ms(t2, t3));
    total_times.push_back(ms(t0, t3));
  }
  write_floats(o.output / "input.f32", input.data(), input.size());
  write_floats(o.output / "output.f32", outputs[0].GetTensorData<float>(),
               size_t(output_shape[1]) * 6);
  std::string profile_path;
  if (o.profile)
    profile_path = session.EndProfilingAllocated(allocator).get();
  const auto pre = summarize(preprocess_times),
             infer = summarize(inference_times),
             post = summarize(postprocess_times),
             total = summarize(total_times);
  std::ofstream json(o.output / "result.json");
  if (!json)
    throw std::runtime_error("Cannot write result.json");
  json << std::fixed << std::setprecision(6);
  json << "{\n\"provider\":\"" << o.provider << "\",\"runtime_version\":\""
       << runtime.version << "\",\n";
  json << "\"model\":\"" << escape(fs::absolute(o.model).u8string())
       << "\",\"image\":\"" << escape(fs::absolute(o.image).u8string())
       << "\",\n";
  json << "\"image_size\":[" << image.width << "," << image.height
       << "],\"input_shape\":[1,3," << side << "," << side
       << "],\"output_shape\":[1," << output_shape[1] << ",6],\n";
  json << "\"letterbox\":{\"scale\":" << std::setprecision(12)
       << transform.scale << ",\"left\":" << transform.left
       << ",\"top\":" << transform.top
       << ",\"resized_width\":" << transform.resized_w
       << ",\"resized_height\":" << transform.resized_h << "},\n"
       << std::setprecision(6);
  json << "\"threshold\":" << o.threshold << ",\"iterations\":" << o.iterations
       << ",\"warmup\":" << o.warmup << ",\"threads\":" << o.threads
       << ",\"initialization_ms\":" << initialization_ms << ",\n";
  json << "\"timing_scope\":\"in-memory image preprocessing + synchronous ORT "
          "Run + decoding; excludes file IO, image decoding, drawing, session "
          "initialization\",\n";
  json << "\"timings_ms\":{\"preprocess\":";
  json_stats(json, pre);
  json << ",\"inference\":";
  json_stats(json, infer);
  json << ",\"postprocess\":";
  json_stats(json, post);
  json << ",\"pipeline\":";
  json_stats(json, total);
  json << "},\n";
  json << "\"profile_path\":\"" << escape(profile_path)
       << "\",\"detections\":[";
  std::ofstream csv(o.output / "detections.csv");
  if (!csv)
    throw std::runtime_error("Cannot write detections.csv");
  csv << "class_id,label,confidence,x1,y1,x2,y2\n"
      << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto &d = detections[i];
    if (i)
      json << ",";
    json << "{\"class_id\":" << d.class_id << ",\"label\":\""
         << escape(labels[d.class_id]) << "\",\"confidence\":" << d.confidence
         << ",\"box\":[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2
         << "]}";
    csv << d.class_id << ",\"" << labels[d.class_id] << "\"," << d.confidence
        << "," << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "\n";
    std::cout << labels[d.class_id] << " " << d.confidence << " [" << d.x1
              << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]\n";
  }
  json << "]\n}\n";
  annotate(image, detections);
  const fs::path png = o.output / "annotated.png";
  std::ofstream png_stream(png, std::ios::binary);
  if (!png_stream)
    throw std::runtime_error("Cannot write annotated.png");
  const auto callback = [](void *context, void *data, int size) {
    static_cast<std::ofstream *>(context)->write(static_cast<char *>(data),
                                                 size);
  };
  if (!stbi_write_png_to_func(callback, &png_stream, image.width, image.height,
                              3, image.rgb.data(), image.width * 3) ||
      !png_stream)
    throw std::runtime_error("PNG write failed");
  std::cout << "Provider=" << o.provider << " ORT=" << runtime.version
            << " detections=" << detections.size()
            << " inference_mean_ms=" << infer.mean
            << " pipeline_p95_ms=" << total.p95 << "\n";
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
    return run(args);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

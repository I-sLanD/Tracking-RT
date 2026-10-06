#pragma once
#include "native.hpp"

class NativeModel {
public:
  Ort::Session session{nullptr};
  std::vector<float> input, output;
  std::vector<int64_t> input_shape, output_shape;
  NativeModel(Ort::Env &env, const Runtime &runtime, const fs::path &model,
              const std::string &provider, int threads, int device,
              const fs::path &profile = {}) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (provider == "dml") {
#ifdef _WIN32
      using AppendDml = OrtStatus *(ORT_API_CALL *)(OrtSessionOptions *, int);
      const auto append = reinterpret_cast<AppendDml>(
          runtime.symbol("OrtSessionOptionsAppendExecutionProvider_DML"));
      if (!append)
        throw std::runtime_error("DirectML entry point missing");
      options.DisableMemPattern();
      Ort::ThrowOnError(append(options, device));
#else
      throw std::runtime_error("DirectML requires Windows");
#endif
    } else if (provider != "cpu")
      throw std::runtime_error("Provider must be cpu or dml");
    if (!profile.empty())
      options.EnableProfiling(profile.c_str());
    session = Ort::Session(env, fs::absolute(model).c_str(), options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
      throw std::runtime_error("Expected single-input/output model");
    const auto input_type = session.GetInputTypeInfo(0);
    const auto output_type = session.GetOutputTypeInfo(0);
    const auto in = input_type.GetTensorTypeAndShapeInfo();
    const auto out = output_type.GetTensorTypeAndShapeInfo();
    input_shape = in.GetShape();
    output_shape = out.GetShape();
    if (in.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        out.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("Expected FP32 models");
    const auto count = [](const std::vector<int64_t> &shape) {
      size_t total = 1;
      for (auto dimension : shape) {
        if (dimension <= 0 || dimension > 100000000 ||
            total > 100000000 / size_t(dimension))
          throw std::runtime_error("Expected bounded static model shape");
        total *= size_t(dimension);
      }
      return total;
    };
    input.resize(count(input_shape));
    output.resize(count(output_shape));
    auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensor_ =
        Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                        input_shape.data(), input_shape.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(
        memory, output.data(), output.size(), output_shape.data(),
        output_shape.size());
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session.GetInputNameAllocated(0, allocator).get();
    output_name_ = session.GetOutputNameAllocated(0, allocator).get();
  }
  NativeModel(const NativeModel &) = delete;
  NativeModel &operator=(const NativeModel &) = delete;
  void run() {
    const char *ins[] = {input_name_.c_str()};
    const char *outs[] = {output_name_.c_str()};
    session.Run(Ort::RunOptions{nullptr}, ins, &input_tensor_, 1, outs,
                &output_tensor_, 1);
  }
  std::string end_profile() {
    Ort::AllocatorWithDefaultOptions allocator;
    return session.EndProfilingAllocated(allocator).get();
  }

private:
  Ort::Value input_tensor_{nullptr}, output_tensor_{nullptr};
  std::string input_name_, output_name_;
};

std::array<int, 4> reid_crop(const Image &image, const Detection &d,
                             std::vector<float> &tensor) {
  const int left = std::clamp(int(std::floor(d.x1)), 0, image.width - 1);
  const int top = std::clamp(int(std::floor(d.y1)), 0, image.height - 1);
  const int right = std::clamp(int(std::ceil(d.x2)), left + 1, image.width);
  const int bottom = std::clamp(int(std::ceil(d.y2)), top + 1, image.height);
  constexpr int w = 128, h = 256, plane = w * h;
  constexpr std::array<float, 3> mean{0.485f, 0.456f, 0.406f},
      stddev{0.229f, 0.224f, 0.225f};
  for (int y = 0; y < h; ++y) {
    const double sy = std::clamp((y + 0.5) * (bottom - top) / h - 0.5, 0.0,
                                 double(bottom - top - 1));
    const int y0 = int(sy), y1 = std::min(y0 + 1, bottom - top - 1);
    const double fy = sy - y0;
    for (int x = 0; x < w; ++x) {
      const double sx = std::clamp((x + 0.5) * (right - left) / w - 0.5, 0.0,
                                   double(right - left - 1));
      const int x0 = int(sx), x1 = std::min(x0 + 1, right - left - 1);
      const double fx = sx - x0;
      for (int c = 0; c < 3; ++c) {
        const auto at = [&](int xx, int yy) {
          return image
              .rgb[(size_t(yy + top) * image.width + xx + left) * 3 + c];
        };
        const double value =
            (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x1, y0)) +
            fy * ((1 - fx) * at(x0, y1) + fx * at(x1, y1));
        tensor[size_t(c) * plane + y * w + x] =
            (float(std::lround(value)) / 255.0f - mean[c]) / stddev[c];
      }
    }
  }
  return {left, top, right, bottom};
}
std::vector<float> normalized_embedding(const std::vector<float> &output) {
  double norm = 0;
  for (float v : output) {
    if (!std::isfinite(v))
      throw std::runtime_error("Non-finite ReID output");
    norm += double(v) * v;
  }
  if (norm < 1e-12)
    throw std::runtime_error("Zero ReID embedding");
  std::vector<float> result(output.size());
  const double inverse = 1 / std::sqrt(norm);
  for (size_t i = 0; i < output.size(); ++i)
    result[i] = float(output[i] * inverse);
  return result;
}

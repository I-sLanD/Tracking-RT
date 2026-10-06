#pragma once
// Shared native image, runtime, and YOLO decoding utilities.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
std::string escape(const std::string &s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '\\' || c == '"') {
      out += '\\';
      out += char(c);
    } else if (c == '\n')
      out += "\\n";
    else if (c == '\r')
      out += "\\r";
    else if (c == '\t')
      out += "\\t";
    else if (c < 32)
      throw std::runtime_error("Control character in JSON text");
    else
      out += char(c);
  }
  return out;
}
std::vector<unsigned char> read_bytes(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot read: " + p.u8string());
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
void write_floats(const fs::path &p, const float *data, size_t count) {
  std::ofstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot write: " + p.u8string());
  f.write(reinterpret_cast<const char *>(data),
          std::streamsize(count * sizeof(float)));
  if (!f)
    throw std::runtime_error("Float dump failed");
}

class Runtime {
public:
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
#ifdef _WIN32
  HMODULE handle = nullptr;
#else
  void *handle = nullptr;
#endif
  explicit Runtime(const fs::path &path) {
#ifdef _WIN32
    handle = LoadLibraryExW(fs::absolute(path).c_str(), nullptr,
                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    handle = dlopen(fs::absolute(path).c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (!handle)
      throw std::runtime_error("Cannot load ONNX Runtime: " + path.u8string());
    using GetApiBase = const OrtApiBase *(ORT_API_CALL *)();
    const auto get_api = reinterpret_cast<GetApiBase>(symbol("OrtGetApiBase"));
    if (!get_api)
      throw std::runtime_error("Missing OrtGetApiBase");
    version = get_api()->GetVersionString();
    const auto *api = get_api()->GetApi(ORT_API_VERSION);
    if (!api)
      throw std::runtime_error(
          "ONNX Runtime headers and binary are incompatible");
    Ort::InitApi(api);
  }
  ~Runtime() {
#ifdef _WIN32
    if (handle)
      FreeLibrary(handle);
#else
    if (handle)
      dlclose(handle);
#endif
  }
  void *symbol(const char *name) const {
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(handle, name));
#else
    return dlsym(handle, name);
#endif
  }
  std::string version;
};

struct Image {
  int width, height;
  std::vector<unsigned char> rgb;
};
Image load_image(const fs::path &path) {
  auto bytes = read_bytes(path);
  if (bytes.size() > size_t(INT32_MAX))
    throw std::runtime_error("Image file too large");
  int w = 0, h = 0, channels = 0;
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
      stbi_load_from_memory(bytes.data(), int(bytes.size()), &w, &h, &channels,
                            3),
      stbi_image_free);
  if (!pixels || w <= 0 || h <= 0)
    throw std::runtime_error("Image decode failed");
  return {w, h,
          std::vector<unsigned char>(pixels.get(),
                                     pixels.get() + size_t(w) * h * 3)};
}
struct Transform {
  double scale;
  int left, top, resized_w, resized_h;
};
Transform letterbox(const Image &image, int side, std::vector<float> &tensor) {
  const double r =
      std::min(double(side) / image.width, double(side) / image.height);
  const int rw = int(std::nearbyint(image.width * r));
  const int rh = int(std::nearbyint(image.height * r));
  const int left = (side - rw) / 2, top = (side - rh) / 2;
  const size_t plane = size_t(side) * side;
  std::fill(tensor.begin(), tensor.end(), 114.0f / 255.0f);
  // Half-pixel bilinear resize, RGB -> contiguous float32 NCHW.
  for (int y = 0; y < rh; ++y) {
    const double sy = std::clamp((y + 0.5) * image.height / rh - 0.5, 0.0,
                                 double(image.height - 1));
    const int y0 = int(sy), y1 = std::min(y0 + 1, image.height - 1);
    const double fy = sy - y0;
    for (int x = 0; x < rw; ++x) {
      const double sx = std::clamp((x + 0.5) * image.width / rw - 0.5, 0.0,
                                   double(image.width - 1));
      const int x0 = int(sx), x1 = std::min(x0 + 1, image.width - 1);
      const double fx = sx - x0;
      for (int c = 0; c < 3; ++c) {
        const auto at = [&](int xx, int yy) {
          return image.rgb[(size_t(yy) * image.width + xx) * 3 + c];
        };
        const double value =
            (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x1, y0)) +
            fy * ((1 - fx) * at(x0, y1) + fx * at(x1, y1));
        tensor[size_t(c) * plane + size_t(y + top) * side + x + left] =
            float(std::lround(value)) / 255.0f;
      }
    }
  }
  return {r, left, top, rw, rh};
}
struct Detection {
  float x1, y1, x2, y2, confidence;
  int class_id;
};
std::vector<Detection> decode(const float *output, size_t rows, float threshold,
                              const Transform &t, const Image &image,
                              size_t classes) {
  std::vector<Detection> result;
  for (size_t i = 0; i < rows; ++i) {
    const float *p = output + i * 6;
    if (!std::all_of(p, p + 6, [](float v) { return std::isfinite(v); }))
      throw std::runtime_error("Non-finite model output");
    if (p[4] < threshold)
      continue;
    const int class_id = int(std::lround(p[5]));
    if (class_id < 0 || size_t(class_id) >= classes ||
        std::abs(p[5] - class_id) > 0.01f)
      throw std::runtime_error("Invalid class index; check export head/layout");
    const auto x = [&](float v) {
      return float(
          std::clamp((v - t.left) / t.scale, 0.0, double(image.width)));
    };
    const auto y = [&](float v) {
      return float(
          std::clamp((v - t.top) / t.scale, 0.0, double(image.height)));
    };
    Detection d{x(p[0]), y(p[1]), x(p[2]), y(p[3]), p[4], class_id};
    if (d.x2 > d.x1 && d.y2 > d.y1)
      result.push_back(d);
  }
  // The selected export is end-to-end [x1,y1,x2,y2,score,class_id]; no second
  // NMS.
  return result;
}
void annotate(Image &image, const std::vector<Detection> &detections) {
  for (const auto &d : detections) {
    const std::array<unsigned char, 3> color =
        d.class_id == 0 ? std::array<unsigned char, 3>{30, 220, 90}
                        : std::array<unsigned char, 3>{250, 130, 25};
    const int x1 = std::clamp(int(std::lround(d.x1)), 0, image.width - 1);
    const int y1 = std::clamp(int(std::lround(d.y1)), 0, image.height - 1);
    const int x2 = std::clamp(int(std::lround(d.x2)), 0, image.width - 1);
    const int y2 = std::clamp(int(std::lround(d.y2)), 0, image.height - 1);
    const auto paint = [&](int x, int y) {
      if (x < 0 || x >= image.width || y < 0 || y >= image.height)
        return;
      for (int c = 0; c < 3; ++c)
        image.rgb[(size_t(y) * image.width + x) * 3 + c] = color[c];
    };
    for (int k = 0; k < 3; ++k) {
      for (int x = x1; x <= x2; ++x) {
        paint(x, y1 + k);
        paint(x, y2 - k);
      }
      for (int y = y1; y <= y2; ++y) {
        paint(x1 + k, y);
        paint(x2 - k, y);
      }
    }
  }
}
struct Stats {
  double mean, p50, p95, maximum;
};
Stats summarize(std::vector<double> values) {
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  std::sort(values.begin(), values.end());
  const auto percentile = [&](double q) {
    return values[size_t(std::ceil(q * values.size())) - 1];
  };
  return {mean, percentile(0.5), percentile(0.95), values.back()};
}
void json_stats(std::ostream &f, const Stats &s) {
  f << "{\"mean\":" << s.mean << ",\"p50\":" << s.p50 << ",\"p95\":" << s.p95
    << ",\"max\":" << s.maximum << "}";
}

#pragma once

#include <string>
#include <unordered_map>

#include "glass/quant/fp32_quant.hpp"
#include "glass/quant/sq4_quant.hpp"
#include "glass/quant/sq8_quant.hpp"
#include "glass/quant/sq8_sym_quant.hpp"

namespace glass {

enum class QuantizerType { FP32, SQ8, SQ4, SQ8_SYM };

inline std::unordered_map<int, QuantizerType> quantizer_map;

inline int quantizer_map_init = [] {
  quantizer_map[0] = QuantizerType::FP32;
  quantizer_map[1] = QuantizerType::SQ8;
  quantizer_map[2] = QuantizerType::SQ8;
  quantizer_map[3] = QuantizerType::SQ8_SYM;
  return 42;
}();

} // namespace glass

#pragma once

#include "glass/graph.hpp"

namespace glass {

struct Builder {
  virtual void Build(float *data, float* timestamps, int nb,int max_nb,int miss_end=0) = 0;
  virtual Graph<int> GetGraph() = 0;
  virtual ~Builder() = default;
};

} // namespace glass
/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_TLE_ANALYSIS_TILE_DATAFLOW_RESOURCES_H_
#define TRITON_TLE_ANALYSIS_TILE_DATAFLOW_RESOURCES_H_

#include <cstdint>
#include <optional>

namespace mlir {
class ModuleOp;
namespace triton {
class FuncOp;
namespace tle {

struct TileDeviceResources {
  unsigned registersPerSM;
  unsigned sharedBytesPerSM;
  unsigned maxWarpsPerSM;
  unsigned maxCTAsPerSM;
  unsigned sharedReservedBytes = 1024;
};

/// Architectural resources for supported CUDA targets. Unknown targets return
/// nullopt instead of silently inheriting another GPU's occupancy limits.
std::optional<TileDeviceResources> getTileDeviceResources(ModuleOp module);

/// Estimate whole-function registers per thread from live tensor payloads.
/// Constants, register views and recomputable indexing do not create additional
/// payloads. Dead inputs may supply result registers. The maximum default-group
/// and worker-group pressure is used, including values live across each fork.
/// This is a scheduling proxy with a scalar/control reserve, not a prediction of
/// ptxas allocation: backend scheduling, packing and spills can change it.
unsigned estimateTileRegistersPerThread(triton::FuncOp function);

/// Estimate resource-limited resident CTAs. Registers are rounded to 256 per
/// warp and shared storage to 256 bytes, including the per-CTA driver reserve.
/// Assumes the maximum shared-memory carveout and a single-CTA launch. This
/// estimate does not model grid size, clusters, runtime carveout or throughput.
unsigned estimateTileActiveCTAs(const TileDeviceResources &resources,
                                unsigned warps, unsigned registers,
                                uint64_t sharedBytes);

} // namespace tle
} // namespace triton
} // namespace mlir

#endif // TRITON_TLE_ANALYSIS_TILE_DATAFLOW_RESOURCES_H_

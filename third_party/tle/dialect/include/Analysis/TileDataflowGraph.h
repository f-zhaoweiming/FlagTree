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

#ifndef TRITON_TLE_ANALYSIS_TILE_DATAFLOW_GRAPH_H_
#define TRITON_TLE_ANALYSIS_TILE_DATAFLOW_GRAPH_H_

#include "mlir/IR/Block.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::tle {

/// Whether an operation prevents scheduling across its position. Unknown
/// effects, control flow, asynchronous operations and explicit synchronization
/// are boundaries. Known loads/stores instead use the memory-dependence edges.
bool isSchedulingBoundary(Operation *op);

/// Whether an operation can be placed in a different warp-group region when
/// its graph dependencies are respected. Scalar address/shape operations are
/// included as support nodes; callers must separately require useful tile work.
/// This is a necessary condition, not permission to ignore graph edges.
bool isMovableTileOp(Operation *op);

/// A conservative dataflow DAG for one block of tile-level IR.
///
/// Nodes include scalar support operations and the terminator. Edges cover SSA
/// values (including captures of nested regions), memory RAW/WAR/WAW hazards,
/// and scheduling boundaries. Every edge follows the original block order.
/// Unknown pointer provenance may alias: distinct kernel arguments are never
/// assumed disjoint. Only distinct local allocations or proven different
/// address spaces allow a memory-dependence edge to be omitted.
///
/// This is a snapshot: rebuild after moving or replacing operations. Loop
/// recurrences are not represented as backward edges. The loop operation is a
/// boundary, and a graph of its body describes only one iteration.
class TileDataflowGraph {
public:
  explicit TileDataflowGraph(Block &block);

  llvm::ArrayRef<Operation *> getOperations() const { return operations; }
  llvm::ArrayRef<Operation *> getPredecessors(Operation *op) const;
  llvm::ArrayRef<Operation *> getSuccessors(Operation *op) const;

  /// Reflexive reachability for operations in this graph; false for operations
  /// outside the graph. A path is an ordering constraint, not a data transfer.
  /// Reachability is cached lazily per queried source for this immutable graph.
  bool hasPath(Operation *from, Operation *to) const;

private:
  void addEdge(Operation *from, Operation *to);

  llvm::SmallVector<Operation *> operations;
  llvm::DenseMap<Operation *, unsigned> indices;
  llvm::SmallVector<llvm::SmallVector<Operation *>> predecessors;
  llvm::SmallVector<llvm::SmallVector<Operation *>> successors;
  mutable llvm::DenseMap<unsigned, llvm::BitVector> reachableFrom;
};

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_TILE_DATAFLOW_GRAPH_H_

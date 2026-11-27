#pragma once

namespace llvm {
class Function;
}

namespace mlir::triton::xpu {
// The caller guarantees that the broadcast bias is immutable for this launch.
void reuseRowSoftmaxBias(llvm::Function &kernel);
}

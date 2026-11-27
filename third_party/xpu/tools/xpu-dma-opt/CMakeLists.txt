cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(XPUDmaOpt LANGUAGES C CXX)
  find_package(LLVM REQUIRED CONFIG)
endif()
add_executable(xpu-dma-opt xpu-dma-opt.cpp)
target_include_directories(xpu-dma-opt PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_features(xpu-dma-opt PRIVATE cxx_std_17)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(xpu-dma-opt PRIVATE -fno-rtti)
endif()
if(NOT LLVM_ENABLE_EH)
  target_compile_options(xpu-dma-opt PRIVATE -fno-exceptions)
endif()
llvm_map_components_to_libnames(XPU_DMA_LLVM_LIBS Core IRReader Analysis
                               TransformUtils ScalarOpts Passes Support)
target_link_libraries(xpu-dma-opt PRIVATE ${XPU_DMA_LLVM_LIBS})
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_custom_command(TARGET xpu-dma-opt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../backend/bin
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:xpu-dma-opt>
            ${CMAKE_CURRENT_SOURCE_DIR}/../../backend/bin/xpu-dma-opt)
endif()
install(TARGETS xpu-dma-opt RUNTIME DESTINATION bin)

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <stdfloat>
#include <string>
#include <format>
#include <iostream>
#include <fstream>
#include <chrono>

#include "xrt/xrt_kernel.h"
#include "xrt/xrt_device.h"


int main(int argc, char* argv[]) {
  std::cout << "Start Driver" << std::endl << std::endl;

  assert( argc == 2 );
  std::string xdna_version = argv[1];
  assert( xdna_version == "XDNA1" || xdna_version == "XDNA2" );

  // get device handle
  uint32_t device_index = 0;
  xrt::device device = xrt::device( device_index );

  // load the xclbin
  const std::string xclbin_file_name = std::format("build/final_{}.xclbin", xdna_version);
  xrt::xclbin xclbin = xrt::xclbin( xclbin_file_name );
  // register the xclbin
  device.register_xclbin( xclbin );
  // get a hardware context
  const xrt::hw_context hw_context = xrt::hw_context( device, xclbin.get_uuid() );

  // get the kernel from the xclbin
  std::string kernel_idenitfier = "MLIR_AIE";
  auto xkernels = xclbin.get_kernels();
  xrt::xclbin::kernel xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(),
      [kernel_idenitfier]( xrt::xclbin::kernel &k ) {
        auto name = k.get_name();
        return name.rfind( kernel_idenitfier, 0 ) == 0;
      });
  std::string kernel_name = xkernel.get_name();

  // get a kernel handle
  xrt::kernel kernel = xrt::kernel( hw_context, kernel_name );

  // Open file in binary mode
  std::string insts_path = std::format("build/insts_{}.bin", xdna_version);
  std::ifstream insts_file( insts_path, std::ios::binary );
  if ( !insts_file.is_open() ) {
    throw std::runtime_error( "Unable to open instruction file\n" );
  }

  // Get the size of the instruction file
  insts_file.seekg( 0, std::ios::end );
  std::streamsize size_insts = insts_file.tellg();
  insts_file.seekg( 0, std::ios::beg );

  if (size_insts % 4 != 0) {
    throw std::runtime_error( "File size is not a multiple of 4 bytes\n" );
  }
  xrt::bo bo_insts;

  bo_insts = xrt::bo(
    device, size_insts,
    xrt::bo::flags::cacheable, kernel.group_id( 1 )
  );
  void* ptr_insts = bo_insts.map<void*>();

  if (!insts_file.read(reinterpret_cast<char*>(ptr_insts), size_insts)) {
    throw std::runtime_error("Failed to read instruction file\n");
  }

  bo_insts.sync( XCL_BO_SYNC_BO_TO_DEVICE );

  xrt::bo bo_io;
  std::size_t size_io = 4;
  bo_io = xrt::bo(
    device, size_io,
    xrt::bo::flags::host_only, kernel.group_id( 3 )
  );
  bo_io.sync( XCL_BO_SYNC_BO_TO_DEVICE );

  std::float32_t flop_count;
  if ( xdna_version == "XDNA2" ) {
    flop_count = 2 * 64 * 96 * 64;
  } else {
    flop_count = 2 * 32 * 32 * 32;
  }
  std::float32_t kernel_repetions = 1000000;
  flop_count *= kernel_repetions;

  uint32_t opcode = 3;
  xrt::run run;

  std::float32_t npu_time_total = 0;
  std::size_t n_warmups = 3;
  std::size_t n_iterations = 10;

  std::cout << "Warming Up" << std::endl;

  for ( std::size_t i=0; i < n_warmups + n_iterations; i++ ) {

    auto start = std::chrono::high_resolution_clock::now();
    run = kernel( opcode, bo_insts, size_insts, bo_io );

    ert_cmd_state r = run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    if ( r != ERT_CMD_STATE_COMPLETED ) {
      std::string error_state;
      switch (r) {
        case ERT_CMD_STATE_ABORT:
          error_state = "ABORT"; break;
        case ERT_CMD_STATE_ERROR:
          error_state = "ERROR"; break;
        case ERT_CMD_STATE_TIMEOUT:
          error_state = "TIMEOUT"; break;
        case ERT_CMD_STATE_NORESPONSE:
          error_state = "NORESPONSE"; break;
        default:
          error_state = "UNKOWN";
      }

      throw std::runtime_error(
        "Kernel did not complete. Returned status(" +  std::to_string(r) + "): "
        + error_state + "\n"
      );
    }

    if ( i < n_warmups ) {
      std::cout << i+1 << "/" << n_warmups << std::endl;
    }

    if ( i == n_warmups - 1 ) {
      std::cout << std::endl;
      std::cout << "Starting Measurements" << std::endl;
      std::cout << std::endl;
      std::cout << "| Iteration | Cur. Time (us) | Cur. Perf. (GFLOPS) |" << std::endl;
      std::cout << "|-----------|----------------|---------------------|" << std::endl;
    }

    if ( i >= n_warmups ) {
      std::float32_t npu_time =
          std::chrono::duration_cast<std::chrono::microseconds>( stop - start )
              .count();

      npu_time_total += npu_time;

      std::cout <<  "| " << std::setw(9)  << i - n_warmups
                << " | " << std::setw(14) << npu_time
                << " | " << std::setw(19) << flop_count / ( 1000 * npu_time )
                << " |"  << std::endl;
    }
  }
  std::cout << std::endl;
  std::cout << "Finished Measurements" << std::endl;
  std::cout << std::endl;

  std::float32_t npu_time_avg = npu_time_total / n_iterations;
  std::cout << "Avg. Time: " << npu_time_avg << "us" << std::endl;

  std::float32_t gflops = flop_count / ( 1000 * npu_time_avg );
  std::cout << "Avg. Performance: " << gflops << " GFLOPS" << std::endl;

  std::cout << std::endl << "End Driver" << std::endl;
  return EXIT_SUCCESS;
}

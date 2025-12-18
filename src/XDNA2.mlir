module {
  aie.device(npu2) {
    %shim_tile_0_0 = aie.tile(0, 0)
    %comp_tile_0_2 = aie.tile(0, 2)
    %buffer_in0    = aie.buffer(%comp_tile_0_2) {address = 1024  : i32, sym_name = "buffer_in0"}   : memref<6912xi8>
    %buffer_in1    = aie.buffer(%comp_tile_0_2) {address = 25856 : i32, sym_name = "buffer_in1"}   : memref<6912xi8>
    %buffer_out_0  = aie.buffer(%comp_tile_0_2) {address = 40960 : i32, sym_name = "buffer_out_0"} : memref<8192xi8>
    %buffer_out_1  = aie.buffer(%comp_tile_0_2) {address = 49152 : i32, sym_name = "buffer_out_1"} : memref<8192xi8>
    func.func private @tensor_kernel_64x96x64_bfp16_bfp16_fp32(memref<6912xi8>, memref<6912xi8>, memref<8192xi8>, memref<8192xi8>)
    aie.objectfifo @object_fifo_in(%shim_tile_0_0, {%comp_tile_0_2}, 1 : i32) : !aie.objectfifo<memref<4xi8>>
    aie.objectfifo @object_fifo_out(%comp_tile_0_2, {%shim_tile_0_0}, 1 : i32) : !aie.objectfifo<memref<4xi8>>
    %core_0_2 = aie.core(%comp_tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @object_fifo_in(Consume, 1) : !aie.objectfifosubview<memref<4xi8>>
        %2 = aie.objectfifo.acquire @object_fifo_out(Produce, 1) : !aie.objectfifosubview<memref<4xi8>>
        %c1000000 = arith.constant 1000000 : index
        scf.for %arg1 = %c0 to %c1000000 step %c1 {
          func.call @tensor_kernel_64x96x64_bfp16_bfp16_fp32(%buffer_in0, %buffer_in1, %buffer_out_0, %buffer_out_1) : (memref<6912xi8>, memref<6912xi8>, memref<8192xi8>, memref<8192xi8>) -> ()
        }
        aie.objectfifo.release @object_fifo_out(Produce, 1)
        aie.objectfifo.release @object_fifo_in(Consume, 1)
      }
      aie.end
    } {link_with = "tensor_kernel_64x96x64_bfp16_bfp16_fp32.o"}
    aiex.runtime_sequence(%arg0: memref<4xi8>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]) {id = 0 : i64, metadata = @object_fifo_in} : memref<4xi8>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]) {id = 1 : i64, metadata = @object_fifo_out} : memref<4xi8>
      aiex.npu.dma_wait {symbol = @object_fifo_out}
    }
  }
}

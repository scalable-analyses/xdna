#include <aie_api/aie.hpp>

template <typename T_in, typename T_out,
          unsigned r, unsigned s, unsigned t>
inline static void bf16_mac_template( T_in const * __restrict ptr_in0,
                                      T_in const * __restrict ptr_in1,
                                      T_out      * __restrict ptr_out ) {
  // define matrix multiplication operation
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accfloat>;

  // define vectors
  aie::vector<T_in,  MMUL::size_A> mat_in0;
  aie::vector<T_in,  MMUL::size_B> mat_in1;
  aie::vector<T_out, MMUL::size_C> mat_out;

  // load data
  mat_in0 = aie::load_v<MMUL::size_A>(ptr_in0);
  mat_in1 = aie::load_v<MMUL::size_B>(ptr_in1);
  mat_out = aie::load_v<MMUL::size_C>(ptr_out);

  // declare accumulator
  MMUL mm_out(mat_out);

  // perform matrix multiplication
  mm_out.mac(mat_in0, mat_in1);

  // store accumulator
  aie::store_v(ptr_out, mm_out.template to_vector<T_out>());

  return;
}

// instantiate template
extern "C" {
  void bfp16_emu_mac( bfloat16 const * ptr_in0,
                      bfloat16 const * ptr_in1,
                      float          * ptr_out ) {
    bf16_mac_template<bfloat16, float, 8, 8, 8>(ptr_in0, ptr_in1, ptr_out);
  }
}

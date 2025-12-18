  .globl tensor_kernel_32x32x32_bf16_bf16_fp32
  .p2align 4
  .type tensor_kernel_32x32x32_bf16_bf16_fp32,@function
tensor_kernel_32x32x32_bf16_bf16_fp32:

                                                                                                                                                 // r,s,t=4,8,4; m,k,n=8,4,8(M/r,K/s,N/t)
  nopv                          ; vlda amlh0, [p2, #32] ; vldb wl4, [p1], #32 ; nops                 ; movxm m2, #4*4*4 * 8                     // 4(byte)*4(r)*4(t) * 8(n)                                             // out - 1 row-step
  nopv                          ; vlda amll0, [p2], m2  ; vldb wh4, [p1], #32 ; nops                 ; movx r0,  #28            ; mov p3, p2
  nopv                          ; vlda amhh0, [p2, #32] ; vldb wl0, [p0], #32 ; nops                 ; movxm m7, #2*4*8 * 4 - 32                // 2(byte)*4(r)*8(s) * 4(k) - 32 (half-block)                           // in0 - m-step
  nopv                          ; vlda amhl0, [p2], #64 ; vldb wh0, [p0], m7  ; nops                 ; nopx                     ; mov p4, p2
  nopv                          ; vlda amll1, [p3, #64] ; vldb wl1, [p0], #32 ; nops                 ; movxm m0, #32 - (2*4*8*4)                // 32(half-block) - (m7+32)                                             // in0 - k-step
  nopv                          ; vlda amlh1, [p3, #96] ; vldb wh1, [p0], m0  ; padds [p3], #128     ; nopx                     ; mov p5, p3
  nopv                          ; vlda amhl1, [p2], #32 ; vldb wl5, [p1], #32 ; nops                 ; movxm m1, #2*8*4 * 8 - 7 * 32            // 2(byte)*8(s)*4(t) * 8(n) - 7(blocking in n *2 - 1) * 32(half-block)  // in1 - k-step
  nopv                          ; vlda amhh1, [p2], #32 ; vldb wh5, [p1], #32 ; nops                 ; movx r1,  #0             ; mov r2, #8/4  // 8(n)/4(blocking in n)
  nopv                          ; vlda amll2, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; movxm r3, #32 - 2*4*8 * 2 * 4            // 32(half-block) - 2(byte)*4(r)*8(s) * 2(blocking in m) * 4(k)         // in0 - n-step
  nopv                          ; vlda amlh2, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; movxm r4, #32                            // 32(half-block)                                                       // in0 - m-step // out - n-step
  vmac.f bml0, bml0, x0, x4, r0 ; vlda amhl2, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; movxm r5, #32 - 2*8*4 * 8 * (4-1)        // 32(half-block) - 2(byte)*8(s)*4(t) * 8(n) * (k-1)                    // in1 - n-step
  nopv                          ; vlda amhh2, [p2], #32 ; vldb wh7, [p1], m1  ; nops                 ; movxm r6, #32 - 2*8*4 * 8 * 4            // 32(half-block) - 2(byte)*8(s)*4(t) * 8(n) * 2(4)                     // in1 - m-step
  vmac.f bmh0, bmh0, x1, x4, r0 ; vlda amll3, [p3], #32 ; nopb                ; nops                 ; movxm r7, #32 + 4*4*4 * 8                // 32(half-block) + 4(byte)*4(r)*4(t) * 8(n)                            // out - m-step
  nopv                          ; vlda amlh3, [p3], #32 ; nopb                ; nops                 ; add r1,  r1,  #1         ; nopm
  vmac.f bml1, bml1, x0, x5, r0 ; vlda amhl3, [p2], #32 ; nopb                ; nops                 ; ltu r27, r1,  r2         ; nopm
  nopv                          ; vlda amhh3, [p2], #32 ; nopb                ; nops                 ; sel.nez r28, r4, r7, r27 ; nopm
  vmac.f bmh1, bmh1, x1, x5, r0 ; vlda wl2,   [p0], #32 ; vldb wl4, [p1], #32 ; nops                 ; sel.nez r28, r5, r6, r27 ; mov m5, r28
  nopv                          ; vlda wh2,   [p0], m7  ; vldb wh4, [p1], #32 ; nops                 ; sel.nez r28, r3, r4, r27 ; mov m4, r28
  vmac.f bml2, bml2, x0, x6, r0 ; vlda wl3,   [p0], #32 ; vldb wl5, [p1], #32 ; nops                 ; mul r1,  r1,  r27        ; mov m3, r28
  nopv                          ; vlda wh3,   [p0], m0  ; vldb wh5, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh2, bmh2, x1, x6, r0 ; vlda amll4, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  nopv                          ; vlda amlh4, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml3, bml3, x0, x7, r0 ; vlda amhl4, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; movxm ls, #.l_start
  nopv                          ; vlda amhh4, [p2], #32 ; vldb wh7, [p1], m1  ; nops                 ; movxm le, #.l_end
  vmac.f bmh3, bmh3, x1, x7, r0 ; vlda wl0,   [p0], #32 ; vldb wl4, [p1], #32 ; nops                 ; movxm lc, #3                            // (8(m)/2(blocking in m) * 8(n)/4(blocking in n) - 2(warmup + cool-down)) /2(iterations in loop)

// k=8
  vmac.f bml0, bml0, x2, x4, r0 ; vlda wh0,   [p0], m7  ; vldb wh4, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh0, bmh0, x3, x4, r0 ; vlda wl1,   [p0], #32 ; vldb wl5, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml1, bml1, x2, x5, r0 ; vlda wh1,   [p0], m0  ; vldb wh5, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh1, bmh1, x3, x5, r0 ; vlda amll5, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml2, bml2, x2, x6, r0 ; vlda amlh5, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh2, bmh2, x3, x6, r0 ; vlda amhl5, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml3, bml3, x2, x7, r0 ; vlda amhh5, [p2], #32 ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
  vmac.f bmh3, bmh3, x3, x7, r0 ; vlda wl2,   [p0], #32 ; vldb wl4, [p1], #32 ; nops                 ; nopx                     ; nopm

// k=16
  vmac.f bml0, bml0, x0, x4, r0 ; vlda wh2,   [p0], m7  ; vldb wh4, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh0, bmh0, x1, x4, r0 ; vlda wl3,   [p0], #32 ; vldb wl5, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml1, bml1, x0, x5, r0 ; vlda wh3,   [p0], m3  ; vldb wh5, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh1, bmh1, x1, x5, r0 ; vlda amll6, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml2, bml2, x0, x6, r0 ; vlda amlh6, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh2, bmh2, x1, x6, r0 ; vlda amhl6, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml3, bml3, x0, x7, r0 ; vlda amhh6, [p2], #32 ; vldb wh7, [p1], m4  ; nops                 ; nopx                     ; nopm
  vmac.f bmh3, bmh3, x1, x7, r0 ; vlda wl0,   [p0], #32 ; vldb wl4, [p1], #32 ; nops                 ; nopx                     ; nopm

// k=24
  vmac.f bml0, bml0, x2, x4, r0 ; vlda wh0,   [p0], m7  ; vldb wh4, [p1], #32 ; nops                 ; add r1,  r1,  #1         ; nopm
  vmac.f bmh0, bmh0, x3, x4, r0 ; vlda wl1,   [p0], #32 ; vldb wl5, [p1], #32 ; nops                 ; ltu r27, r1,  r2         ; nopm
  vmac.f bml1, bml1, x2, x5, r0 ; vlda wh1,   [p0], m0  ; vldb wh5, [p1], #32 ; nops                 ; sel.nez r28, r4, r7, r27 ; mov m6, m5
  vmac.f bmh1, bmh1, x3, x5, r0 ; vlda amll7, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; sel.nez r28, r5, r6, r27 ; mov m5, r28
  vmac.f bml2, bml2, x2, x6, r0 ; vlda amlh7, [p3], m5  ; vldb wh6, [p1], #32 ; nops                 ; sel.nez r28, r3, r4, r27 ; mov m4, r28
  vmac.f bmh2, bmh2, x3, x6, r0 ; vlda amhl7, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; mul r1,  r1,  r27        ; mov m3, r28
  vmac.f bml3, bml3, x2, x7, r0 ; vlda amhh7, [p2], m5  ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
  vmac.f bmh3, bmh3, x3, x7, r0 ; vlda wl2,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll0, [p5], #32 ; nopx                     ; nopm
// k=32

.p2align 4
.l_start:
// k=0
  vmac.f bml4, bml4, x0, x4, r0 ; vlda wh2,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh0, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh4, bmh4, x1, x4, r0 ; vlda wl3,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl0, [p4], #32 ; nopx                     ; nopm
  vmac.f bml5, bml5, x0, x5, r0 ; vlda wh3,   [p0], m0  ; vldb wh5, [p1], #32 ; vst amhh0, [p4], #32 ; nopx                     ; nopm
  vmac.f bmh5, bmh5, x1, x5, r0 ; vlda amll0, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml6, bml6, x0, x6, r0 ; vlda amlh0, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh6, bmh6, x1, x6, r0 ; vlda amhl0, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml7, bml7, x0, x7, r0 ; vlda amhh0, [p2], #32 ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
  vmac.f bmh7, bmh7, x1, x7, r0 ; vlda wl0,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll1, [p5], #32 ; nopx                     ; nopm

// k=8
  vmac.f bml4, bml4, x2, x4, r0 ; vlda wh0,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh1, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh4, bmh4, x3, x4, r0 ; vlda wl1,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl1, [p4], #32 ; nopx                     ; nopm
  vmac.f bml5, bml5, x2, x5, r0 ; vlda wh1,   [p0], m0  ; vldb wh5, [p1], #32 ; vst amhh1, [p4], #32 ; nopx                     ; nopm
  vmac.f bmh5, bmh5, x3, x5, r0 ; vlda amll1, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml6, bml6, x2, x6, r0 ; vlda amlh1, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh6, bmh6, x3, x6, r0 ; vlda amhl1, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml7, bml7, x2, x7, r0 ; vlda amhh1, [p2], #32 ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
  vmac.f bmh7, bmh7, x3, x7, r0 ; vlda wl2,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll2, [p5], #32 ; nopx                     ; nopm

// k=16
  vmac.f bml4, bml4, x0, x4, r0 ; vlda wh2,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh2, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh4, bmh4, x1, x4, r0 ; vlda wl3,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl2, [p4], #32 ; nopx                     ; nopm
  vmac.f bml5, bml5, x0, x5, r0 ; vlda wh3,   [p0], m3  ; vldb wh5, [p1], #32 ; vst amhh2, [p4], #32 ; nopx                     ; nopm
  vmac.f bmh5, bmh5, x1, x5, r0 ; vlda amll2, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml6, bml6, x0, x6, r0 ; vlda amlh2, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh6, bmh6, x1, x6, r0 ; vlda amhl2, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml7, bml7, x0, x7, r0 ; vlda amhh2, [p2], #32 ; vldb wh7, [p1], m4  ; nops                 ; nopx                     ; nopm
  vmac.f bmh7, bmh7, x1, x7, r0 ; vlda wl0,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll3, [p5], #32 ; nopx                     ; nopm

// k=24
  vmac.f bml4, bml4, x2, x4, r0 ; vlda wh0,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh3, [p5], m6  ; add r1,  r1,  #1         ; nopm
  vmac.f bmh4, bmh4, x3, x4, r0 ; vlda wl1,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl3, [p4], #32 ; ltu r27, r1,  r2         ; nopm
  vmac.f bml5, bml5, x2, x5, r0 ; vlda wh1,   [p0], m0  ; vldb wh5, [p1], #32 ; vst amhh3, [p4], m6  ; sel.nez r28, r4, r7, r27 ; mov m6, m5
  vmac.f bmh5, bmh5, x3, x5, r0 ; vlda amll3, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; sel.nez r28, r5, r6, r27 ; mov m5, r28
  vmac.f bml6, bml6, x2, x6, r0 ; vlda amlh3, [p3], m5  ; vldb wh6, [p1], #32 ; nops                 ; sel.nez r28, r3, r4, r27 ; mov m4, r28
  vmac.f bmh6, bmh6, x3, x6, r0 ; vlda amhl3, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; mul r1,  r1,  r27        ; mov m3, r28
  vmac.f bml7, bml7, x2, x7, r0 ; vlda amhh3, [p2], m5  ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
  vmac.f bmh7, bmh7, x3, x7, r0 ; vlda wl2,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll4, [p5], #32 ; nopx                     ; nopm
// k=32

// k=0
  vmac.f bml0, bml0, x0, x4, r0 ; vlda wh2,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh4, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh0, bmh0, x1, x4, r0 ; vlda wl3,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl4, [p4], #32 ; nopx                     ; nopm
  vmac.f bml1, bml1, x0, x5, r0 ; vlda wh3,   [p0], m0  ; vldb wh5, [p1], #32 ; vst amhh4, [p4], #32 ; nopx                     ; nopm
  vmac.f bmh1, bmh1, x1, x5, r0 ; vlda amll4, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml2, bml2, x0, x6, r0 ; vlda amlh4, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh2, bmh2, x1, x6, r0 ; vlda amhl4, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml3, bml3, x0, x7, r0 ; vlda amhh4, [p2], #32 ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
  vmac.f bmh3, bmh3, x1, x7, r0 ; vlda wl0,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll5, [p5], #32 ; nopx                     ; nopm

// k=8
  vmac.f bml0, bml0, x2, x4, r0 ; vlda wh0,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh5, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh0, bmh0, x3, x4, r0 ; vlda wl1,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl5, [p4], #32 ; nopx                     ; nopm
  vmac.f bml1, bml1, x2, x5, r0 ; vlda wh1,   [p0], m0  ; vldb wh5, [p1], #32 ; vst amhh5, [p4], #32 ; nopx                     ; nopm
  vmac.f bmh1, bmh1, x3, x5, r0 ; vlda amll5, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml2, bml2, x2, x6, r0 ; vlda amlh5, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh2, bmh2, x3, x6, r0 ; vlda amhl5, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml3, bml3, x2, x7, r0 ; vlda amhh5, [p2], #32 ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
  vmac.f bmh3, bmh3, x3, x7, r0 ; vlda wl2,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll6, [p5], #32 ; nopx                     ; nopm

// k=16
  vmac.f bml0, bml0, x0, x4, r0 ; vlda wh2,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh6, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh0, bmh0, x1, x4, r0 ; vlda wl3,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl6, [p4], #32 ; nopx                     ; nopm
  vmac.f bml1, bml1, x0, x5, r0 ; vlda wh3,   [p0], m3  ; vldb wh5, [p1], #32 ; vst amhh6, [p4], #32 ; nopx                     ; nopm
  vmac.f bmh1, bmh1, x1, x5, r0 ; vlda amll6, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml2, bml2, x0, x6, r0 ; vlda amlh6, [p3], #32 ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh2, bmh2, x1, x6, r0 ; vlda amhl6, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml3, bml3, x0, x7, r0 ; vlda amhh6, [p2], #32 ; vldb wh7, [p1], m4  ; nops                 ; nopx                     ; nopm
  vmac.f bmh3, bmh3, x1, x7, r0 ; vlda wl0,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll7, [p5], #32 ; nopx                     ; nopm

// k=24
  vmac.f bml0, bml0, x2, x4, r0 ; vlda wh0,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh7, [p5], m6  ; add r1,  r1,  #1         ; nopm
  vmac.f bmh0, bmh0, x3, x4, r0 ; vlda wl1,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl7, [p4], #32 ; ltu r27, r1,  r2         ; nopm
  vmac.f bml1, bml1, x2, x5, r0 ; vlda wh1,   [p0], m0  ; vldb wh5, [p1], #32 ; vst amhh7, [p4], m6  ; sel.nez r28, r4, r7, r27 ; mov m6, m5
  vmac.f bmh1, bmh1, x3, x5, r0 ; vlda amll7, [p3], #32 ; vldb wl6, [p1], #32 ; nops                 ; sel.nez r28, r5, r6, r27 ; mov m5, r28
  vmac.f bml2, bml2, x2, x6, r0 ; vlda amlh7, [p3], m5  ; vldb wh6, [p1], #32 ; nops                 ; sel.nez r28, r3, r4, r27 ; mov m4, r28
  vmac.f bmh2, bmh2, x3, x6, r0 ; vlda amhl7, [p2], #32 ; vldb wl7, [p1], #32 ; nops                 ; mul r1,  r1,  r27        ; mov m3, r28
  vmac.f bml3, bml3, x2, x7, r0 ; vlda amhh7, [p2], m5  ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
.p2align 4
.l_end:
  vmac.f bmh3, bmh3, x3, x7, r0 ; vlda wl2,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll0, [p5], #32 ; nopx                     ; nopm
// k=32

// k=0
  vmac.f bml4, bml4, x0, x4, r0 ; vlda wh2,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh0, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh4, bmh4, x1, x4, r0 ; vlda wl3,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl0, [p4], #32 ; nopx                     ; nopm
  vmac.f bml5, bml5, x0, x5, r0 ; vlda wh3,   [p0], m0  ; vldb wh5, [p1], #32 ; vst amhh0, [p4], #32 ; nopx                     ; nopm
  vmac.f bmh5, bmh5, x1, x5, r0 ; nopa                  ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml6, bml6, x0, x6, r0 ; nopa                  ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh6, bmh6, x1, x6, r0 ; nopa                  ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml7, bml7, x0, x7, r0 ; nopa                  ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
  vmac.f bmh7, bmh7, x1, x7, r0 ; vlda wl0,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll1, [p5], #32 ; nopx                     ; nopm

// k=8
  vmac.f bml4, bml4, x2, x4, r0 ; vlda wh0,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh1, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh4, bmh4, x3, x4, r0 ; vlda wl1,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl1, [p4], #32 ; nopx                     ; nopm
  vmac.f bml5, bml5, x2, x5, r0 ; vlda wh1,   [p0], m0  ; vldb wh5, [p1], #32 ; vst amhh1, [p4], #32 ; nopx                     ; nopm
  vmac.f bmh5, bmh5, x3, x5, r0 ; nopa                  ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml6, bml6, x2, x6, r0 ; nopa                  ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh6, bmh6, x3, x6, r0 ; nopa                  ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml7, bml7, x2, x7, r0 ; nopa                  ; vldb wh7, [p1], m1  ; nops                 ; nopx                     ; nopm
  vmac.f bmh7, bmh7, x3, x7, r0 ; vlda wl2,   [p0], #32 ; vldb wl4, [p1], #32 ; vst amll2, [p5], #32 ; nopx                     ; nopm

// k=16
  vmac.f bml4, bml4, x0, x4, r0 ; vlda wh2,   [p0], m7  ; vldb wh4, [p1], #32 ; vst amlh2, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh4, bmh4, x1, x4, r0 ; vlda wl3,   [p0], #32 ; vldb wl5, [p1], #32 ; vst amhl2, [p4], #32 ; nopx                     ; nopm
  vmac.f bml5, bml5, x0, x5, r0 ; vlda wh3,   [p0, #0]  ; vldb wh5, [p1], #32 ; vst amhh2, [p4], #32 ; nopx                     ; nopm
  vmac.f bmh5, bmh5, x1, x5, r0 ; nopa                  ; vldb wl6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml6, bml6, x0, x6, r0 ; nopa                  ; vldb wh6, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bmh6, bmh6, x1, x6, r0 ; nopa                  ; vldb wl7, [p1], #32 ; nops                 ; nopx                     ; nopm
  vmac.f bml7, bml7, x0, x7, r0 ; nopa                  ; vldb wh7, [p1, #0]  ; nops                 ; nopx                     ; nopm
  vmac.f bmh7, bmh7, x1, x7, r0 ; nopa                  ; nopb                ; vst amll3, [p5], #32 ; nopx                     ; nopm

// k=24
  vmac.f bml4, bml4, x2, x4, r0 ; nopa                  ; nopb                ; vst amlh3, [p5], m6  ; nopx                     ; nopm
  vmac.f bmh4, bmh4, x3, x4, r0 ; nopa                  ; nopb                ; vst amhl3, [p4], #32 ; nopx                     ; nopm
  vmac.f bml5, bml5, x2, x5, r0 ; nopa                  ; nopb                ; vst amhh3, [p4], m6  ; nopx                     ; nopm
  vmac.f bmh5, bmh5, x3, x5, r0 ; nopa                  ; nopb                ; nops                 ; nopx                     ; nopm
  vmac.f bml6, bml6, x2, x6, r0 ; nopa                  ; nopb                ; nops                 ; nopx                     ; nopm
  vmac.f bmh6, bmh6, x3, x6, r0 ; nopa                  ; nopb                ; nops                 ; nopx                     ; nopm
  vmac.f bml7, bml7, x2, x7, r0 ; nopa                  ; nopb                ; vst amll4, [p5], #32 ; nopx                     ; nopm
  vmac.f bmh7, bmh7, x3, x7, r0 ; nopa                  ; nopb                ; vst amlh4, [p5], #32 ; nopx                     ; nopm
// k=32

  nopv                          ; nopa                  ; nopb                ; vst amhl4, [p4], #32 ; nopx                     ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amhh4, [p4], #32 ; nopx                     ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amll5, [p5], #32 ; nopx                     ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amlh5, [p5], #32 ; nopx                     ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amhl5, [p4], #32 ; nopx                     ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amhh5, [p4], #32 ; nopx                     ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amll6, [p5], #32 ; nopx                     ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amlh6, [p5], #32 ; nopx                     ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amhl6, [p4], #32 ; nopx                     ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amhh6, [p4], #32 ; ret lr                   ; nopm
  nopv                          ; nopa                  ; nopb                ; vst amll7, [p5], #32 ; nopx                     ; nopm //  Delay Slot 5
  nopv                          ; nopa                  ; nopb                ; vst amlh7, [p5, #0]  ; nopx                     ; nopm //  Delay Slot 4
  nopv                          ; nopa                  ; nopb                ; vst amhl7, [p4], #32 ; nopx                     ; nopm //  Delay Slot 3
  nopv                          ; nopa                  ; nopb                ; vst amhh7, [p4, #0]  ; nopx                     ; nopm //  Delay Slot 2
  nopv                          ; nopa                  ; nopb                ; nops                 ; nopx                     ; nopm //  Delay Slot 1

.Lfunc_end0:
        .size tensor_kernel_32x32x32_bf16_bf16_fp32, .Lfunc_end0-tensor_kernel_32x32x32_bf16_bf16_fp32

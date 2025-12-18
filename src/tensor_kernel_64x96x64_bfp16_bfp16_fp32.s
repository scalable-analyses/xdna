  .globl tensor_kernel_64x96x64_bfp16_bfp16_fp32
  .p2align 4
  .type tensor_kernel_64x96x64_bfp16_bfp16_fp32,@function
tensor_kernel_64x96x64_bfp16_bfp16_fp32:
  .p2align 4

// k=0
// 0                                                                                                                             //          block   B   k
  nopv                            ; vlda bmll0, [p2, #0  ]            ; vldb x8,  [p3, #0  ]              ; movs p4, p2          ; movxm r4, #-576 / 8 * 12 * 2     // m1       - jumpback in IN0 (m dimension)
  nopv                            ; vlda bmlh0, [p2, #64 ]            ; vldb x9,  [p3, #64 ]              ; movs p5, p3          ; movxm r5, #-576 / 8 * 12 * 2 * 4 // n1 * n2  - jumpback in IN1 (complete tensor)
  nopv                            ; vlda bmhl0, [p2, #128]            ; vldb x10, [p3, #128]              ; padds [p4], #256     ; movx r24, #0   ; mov  r25, #0
  nopv                            ; vlda bmhh0, [p2, #192]            ; vldb x11, [p3, #192]              ; padds [p5], #256     ; movx r0,  #780 ; mov  r1, #0

// 4
  nopv                            ; vlda bmhl1, [p4, #128]            ; vldb x8,  [p5], #64               ; nops                 ; movxm ls, #.l_start
  nopv                            ; vlda bmhh1, [p4, #192]            ; vldb x9,  [p5], #64               ; nops                 ; movxm le, #.l_end
  nopv                            ; vlda bmll1, [p4], #64             ; vldb x10, [p5], #64               ; nops                 ; movxm lc, #14                    // 4(m2) * 4(n2) - 2
  nopv                            ; vlda bmlh1, [p4], #192            ; vldb x11, [p5], #64               ; nops                 ; nopxm

// 8
  nopv                            ; vlda.fill.512 [p0, lf0, r24]      ; vldb.fill.512 [p1, lf1, r25]      ; nops                 ; nopx           ; vmov bmll2, x8
  nopv                            ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmlh2, x9
  nopv                            ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmhl2, x10
  nopv                            ; vlda.pop.576 ex4,  [p0, lf0, r24] ; vldb.pop.576 ex5,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmhh2, x11
  nopv                            ; vlda.pop.576 ex6,  [p0, lf0, r24] ; vldb.pop.576 ex7,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmll3, x8
  nopv                            ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmlh3, x9
  nopv                            ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmhl3, x10
  nopv                            ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmhh3, x11
  nopv                            ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; nopxm

// 17
// k=0
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda bmll2, [p4], #64             ; vldb x4, [p5], #64                ; nops                 ; nopxm
  vmac.f dm0, dm2, ex0,  ex3,  r0 ; vlda bmlh2, [p4], #64             ; vldb x5, [p5], #64                ; nops                 ; movx r3, #1    ; nopm
  vmac.f dm1, dm3, ex2,  ex3,  r0 ; vlda bmll2, [p4], #64             ; vldb x6, [p5], #64                ; nops                 ; movx r2, #4    ; nopm

// k=8
  vmac.f dm0, dm0, ex4,  ex5,  r0 ; vlda bmlh2, [p4], #64             ; vldb x7, [p5], #64                ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex5,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm0, dm0, ex4,  ex7,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex7,  r0 ; vlda.fill.512 [p0, lf0, r24]      ; vldb.fill.512 [p1, lf1, r25]      ; nops                 ; nopxm

// k=16
  vmac.f dm0, dm0, ex8,  ex9,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmll4, x4
  vmac.f dm1, dm1, ex10, ex9,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmlh4, x5
  vmac.f dm0, dm0, ex8,  ex11, r0 ; vlda.pop.576 ex4,  [p0, lf0, r24] ; vldb.pop.576 ex5,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmhl4, x6
  vmac.f dm1, dm1, ex10, ex11, r0 ; vlda.pop.576 ex6,  [p0, lf0, r24] ; vldb.pop.576 ex7,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmhh4, x7

// k=24
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; nops                 ; add r1, r1, r3 ; nopm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; nops                 ; ltu r6, r1, r2 ; nopm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; mul r1, r1, r6 ; nopm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; sub r7, r3, r6 ; nopm

// k=32
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; mul r6, r6, r4 ; nopm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda bmll3, [p4], #64             ; nopb                              ; nops                 ; mul r7, r7, r5 ; nopm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; vlda bmlh3, [p4], #64             ; nopb                              ; nops                 ; nopx           ; mov m0, r6
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; vlda bmhl3, [p4], #64             ; nopb                              ; nops                 ; nopx           ; mov m1, r7

// k=40
  vmac.f dm0, dm0, ex4,  ex5,  r0 ; vlda bmhh3, [p4], #64             ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex5,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm0, dm0, ex4,  ex7,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex7,  r0 ; vlda.fill.512 [p0, lf0, r24]      ; vldb.fill.512 [p1, lf1, r25]      ; nops                 ; nopxm

// k=48
  vmac.f dm0, dm0, ex8,  ex9,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex10, ex9,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm0, dm0, ex8,  ex11, r0 ; vlda.pop.576 ex4,  [p0, lf0, r24] ; vldb.pop.576 ex5,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex10, ex11, r0 ; vlda.pop.576 ex6,  [p0, lf0, r24] ; vldb.pop.576 ex7,  [p1, lf1, r25] ; nops                 ; nopxm

// k=56
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; nopxm

// k=64
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; nopa                              ; vldb x6, [p5], #64                ; nops                 ; nopxm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; nopa                              ; vldb x7, [p5], #64                ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm

// k=72
  vmac.f dm0, dm0, ex4,  ex5,  r0 ; padda [p0], m0                    ; paddb [p1], m1                    ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex5,  r0 ; vlda.fill.512 [p0, lf0, r24]      ; vldb.fill.512 [p1, lf1, r25]      ; nops                 ; nopxm
  vmac.f dm0, dm0, ex4,  ex7,  r0 ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex7,  r0 ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; nops                 ; nopxm

// k=80
  vmac.f dm0, dm0, ex8,  ex9,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex10, ex9,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm0, dm0, ex8,  ex11, r0 ; vlda.pop.576 ex4,  [p0, lf0, r24] ; vldb.pop.576 ex5,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex10, ex11, r0 ; vlda.pop.576 ex6,  [p0, lf0, r24] ; vldb.pop.576 ex7,  [p1, lf1, r25] ; nops                 ; nopxm

// k=88_0
  vmac.f dm2, dm0, ex0,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm3, dm1, ex2,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
// k=96_0

// k=0_0
  vmac.f dm0, dm2, ex8,  ex9,  r0 ; vlda bmhl2, [p5], #64             ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm3, ex10, ex9,  r0 ; vlda bmhh2, [p5], #64             ; nopb                              ; nops                 ; nopxm

.p2align 4
.l_start:
// k=88_1
  vmac.f dm4, dm0, ex0,  ex3,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm2, dm1, ex2,  ex3,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmll2, x6
// k=96_1

// k=0_1
  vmac.f dm0, dm4, ex8,  ex11, r0 ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; vst bmll2, [p2], #64 ; nopx           ; vmov bmlh2, x7
  vmac.f dm1, dm2, ex10, ex11, r0 ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; vst bmlh2, [p2], #64 ; nopxm

// k=8
  vmac.f dm0, dm0, ex4,  ex5,  r0 ; nopa                              ; nopb                              ; vst bmhl2, [p2], #64 ; add r1, r1, r3 ; nopm
  vmac.f dm1, dm1, ex6,  ex5,  r0 ; nopa                              ; nopb                              ; vst bmhh2, [p2], #64 ; ltu r6, r1, r2 ; nopm
  vmac.f dm0, dm0, ex4,  ex7,  r0 ; nopa                              ; nopb                              ; nops                 ; mul r1, r1, r6 ; nopm
  vmac.f dm1, dm1, ex6,  ex7,  r0 ; vlda.fill.512 [p0, lf0, r24]      ; vldb.fill.512 [p1, lf1, r25]      ; nops                 ; sub r7, r3, r6 ; nopm

// k=16
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; vst bmll2, [p3], #64 ; mul r6, r6, r4 ; nopm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; vst bmlh2, [p3], #64 ; mul r7, r7, r5 ; nopm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; vlda.pop.576 ex4,  [p0, lf0, r24] ; vldb.pop.576 ex5,  [p1, lf1, r25] ; vst bmhl2, [p3], #64 ; nopx           ; mov m0, r6
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; vlda.pop.576 ex6,  [p0, lf0, r24] ; vldb.pop.576 ex7,  [p1, lf1, r25] ; vst bmhh2, [p3], #64 ; nopx           ; mov m1, r7

// k=24
  vmac.f dm0, dm0, ex8,  ex9,  r0 ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; vst bmll4, [p3], #64 ; nopxm
  vmac.f dm1, dm1, ex10, ex9,  r0 ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; vst bmlh4, [p3], #64 ; nopxm
  vmac.f dm0, dm0, ex8,  ex11, r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; vst bmhl4, [p3], #64 ; nopxm
  vmac.f dm1, dm1, ex10, ex11, r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; vst bmhh4, [p3], #64 ; nopxm

// k=32
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; nopa                              ; nopb                              ; vst bmll3, [p2], #64 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; nopa                              ; nopb                              ; vst bmlh3, [p2], #64 ; nopxm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; vlda bmll2, [p4], #64             ; vldb x4, [p5], #64                ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; vlda bmlh2, [p4], #64             ; vldb x5, [p5], #64                ; nops                 ; nopxm

// k=40
  vmac.f dm0, dm0, ex4,  ex5,  r0 ; vlda bmhl2, [p4], #64             ; vldb x6, [p5], #64                ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex5,  r0 ; vlda bmhh2, [p4], #64             ; vldb x7, [p5], #64                ; nops                 ; nopxm
  vmac.f dm0, dm0, ex4,  ex7,  r0 ; vlda bmll3, [p4], #64             ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex7,  r0 ; vlda.fill.512 [p0, lf0, r24]      ; vldb.fill.512 [p1, lf1, r25]      ; nops                 ; nopxm

// k=48
  vmac.f dm0, dm0, ex8,  ex9,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; vst bmhl3, [p2], #64 ; nopxm
  vmac.f dm1, dm1, ex10, ex9,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; vst bmhh3, [p2], #64 ; nopxm
  vmac.f dm0, dm0, ex8,  ex11, r0 ; vlda.pop.576 ex4,  [p0, lf0, r24] ; vldb.pop.576 ex5,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmll4, x4
  vmac.f dm1, dm1, ex10, ex11, r0 ; vlda.pop.576 ex6,  [p0, lf0, r24] ; vldb.pop.576 ex7,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmlh4, x5

// k=56
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmhl4, x6
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmhh4, x7
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; nopxm

// k=64
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda bmlh3, [p4], #64             ; vldb x6, [p5], #64                ; nops                 ; nopxm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; vlda bmhl3, [p4], #64             ; vldb x7, [p5], #64                ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; vlda bmhh3, [p4], #64             ; nopb                              ; nops                 ; nopxm

// k=72
  vmac.f dm0, dm0, ex4,  ex5,  r0 ; padda [p0], m0                    ; paddb [p1], m1                    ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex5,  r0 ; vlda.fill.512 [p0, lf0, r24]      ; vldb.fill.512 [p1, lf1, r25]      ; nops                 ; nopxm
  vmac.f dm0, dm0, ex4,  ex7,  r0 ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex7,  r0 ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; nops                 ; nopxm

// k=80
  vmac.f dm0, dm0, ex8,  ex9,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex10, ex9,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm0, dm0, ex8,  ex11, r0 ; vlda.pop.576 ex4,  [p0, lf0, r24] ; vldb.pop.576 ex5,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex10, ex11, r0 ; vlda.pop.576 ex6,  [p0, lf0, r24] ; vldb.pop.576 ex7,  [p1, lf1, r25] ; nops                 ; nopxm

// k=88_0
  vmac.f dm2, dm0, ex0,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm3, dm1, ex2,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
// k=96_0

// k=0_0
  vmac.f dm0, dm2, ex8,  ex9,  r0 ; vlda bmhl2, [p5], #64             ; nopb                              ; nops                 ; nopxm
.p2align 4
.l_end:
  vmac.f dm1, dm3, ex10, ex9,  r0 ; vlda bmhh2, [p5], #64             ; nopb                              ; nops                 ; nopxm

// k=88_1
  vmac.f dm4, dm0, ex0,  ex3,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm2, dm1, ex2,  ex3,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; nopx           ; vmov bmll2, x6
// k=96_1

// k=0_1
  vmac.f dm0, dm4, ex8,  ex11, r0 ; nopa                              ; nopb                              ; vst bmll2, [p2], #64 ; nopx           ; vmov bmlh2, x7
  vmac.f dm1, dm2, ex10, ex11, r0 ; nopa                              ; nopb                              ; vst bmlh2, [p2], #64 ; nopxm

// k=8
  vmac.f dm0, dm0, ex4,  ex5,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; vst bmhl2, [p2], #64 ; nopxm
  vmac.f dm1, dm1, ex6,  ex5,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; vst bmhh2, [p2], #64 ; nopxm
  vmac.f dm0, dm0, ex4,  ex7,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex6,  ex7,  r0 ; vlda.fill.512 [p0, lf0, r24]      ; vldb.fill.512 [p1, lf1, r25]      ; nops                 ; nopxm

// k=16
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; vst bmll2, [p3], #64 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; vst bmlh2, [p3], #64 ; nopxm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; nopa                              ; nopb                              ; vst bmhl2, [p3], #64 ; nopxm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; nopa                              ; nopb                              ; vst bmhh2, [p3], #64 ; nopxm

// k=24
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; vst bmll4, [p3], #64 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; vst bmlh4, [p3], #64 ; nopxm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; nopa                              ; nopb                              ; vst bmhl4, [p3], #64 ; nopxm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; nopa                              ; nopb                              ; vst bmhh4, [p3], #64 ; nopxm

// k=32
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; vst bmll3, [p2], #64 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; vst bmlh3, [p2], #64 ; nopxm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; nopa                              ; nopb                              ; vst bmhl3, [p2], #64 ; nopxm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; nopa                              ; nopb                              ; vst bmhh3, [p2], #64 ; nopxm

// k=40
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; vlda.pop.576 ex4,  [p0, lf0, r24] ; vldb.pop.576 ex5,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; vlda.pop.576 ex6,  [p0, lf0, r24] ; vldb.pop.576 ex7,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm0, dm0, ex0,  ex3,  r0 ; vlda.fill.512 [p0, lf0, r24]      ; vldb.fill.512 [p1, lf1, r25]      ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex3,  r0 ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; nops                 ; nopxm

// k=48
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex2,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm2, dm0, ex0,  ex3,  r0 ; vlda.pop.576 ex0,  [p0, lf0, r24] ; vldb.pop.576 ex1,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm3, dm1, ex2,  ex3,  r0 ; vlda.pop.576 ex2,  [p0, lf0, r24] ; vldb.pop.576 ex3,  [p1, lf1, r25] ; nops                 ; nopxm

// k=56
  vmac.f dm0, dm0, ex4,  ex5,  r0 ; vlda.pop.576 ex4,  [p0, lf0, r24] ; vldb.pop.576 ex5,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex4,  ex7,  r0 ; vlda.pop.576 ex6,  [p0, lf0, r24] ; vldb.pop.576 ex7,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm2, dm2, ex6,  ex5,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm0, dm0, ex8,  ex9,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm

//
  vmac.f dm3, dm3, ex6,  ex7,  r0 ; vlda.pop.576 ex8,  [p0, lf0, r24] ; vldb.pop.576 ex9,  [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm1, dm1, ex8,  ex11, r0 ; vlda.pop.576 ex10, [p0, lf0, r24] ; vldb.pop.576 ex11, [p1, lf1, r25] ; nops                 ; nopxm
  vmac.f dm0, dm0, ex0,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm2, dm2, ex10, ex9,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm

//
  vmac.f dm3, dm3, ex10, ex11, r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm0, dm0, ex4,  ex5,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex0,  ex3,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm2, dm2, ex2,  ex1,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm

//
  vmac.f dm0, dm0, ex8,  ex9,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm1, dm1, ex4,  ex7,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm3, dm3, ex2,  ex3,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm2, dm2, ex6,  ex5,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm

//
  vmac.f dm1, dm1, ex8,  ex11, r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm3, dm3, ex6,  ex7,  r0 ; nopa                              ; nopb                              ; nops                 ; nopxm
  vmac.f dm2, dm2, ex10, ex9,  r0 ; nopa                              ; nopb                              ; vst bmll0, [p2], #64 ; nopxm
  nopv                            ; nopa                              ; nopb                              ; vst bmlh0, [p2], #64 ; nopxm
  vmac.f dm3, dm3, ex10, ex11, r0 ; nopa                              ; nopb                              ; vst bmhl0, [p2], #64 ; nopxm
// k=96

  nopv                            ; nopa                              ; nopb                              ; vst bmhh0, [p2], #64 ; nopxm
  nopv                            ; nopa                              ; nopb                              ; vst bmll1, [p2], #64 ; nopxm
  nopv                            ; nopa                              ; nopb                              ; vst bmlh1, [p2], #64 ; nopxm
  nopv                            ; nopa                              ; nopb                              ; vst bmhl1, [p2], #64 ; nopxm
  nopv                            ; nopa                              ; nopb                              ; vst bmhh1, [p2], #64 ; nopxm
  nopv                            ; nopa                              ; nopb                              ; vst bmll2, [p3], #64 ; nopxm
  nopv                            ; nopa                              ; nopb                              ; vst bmlh2, [p3], #64 ; nopxm
  nopv                            ; nopa                              ; nopb                              ; vst bmhl2, [p3], #64 ; nopxm
  nopv                            ; nopa                              ; nopb                              ; vst bmhh2, [p3], #64 ; ret lr
  nopv                            ; nopa                              ; nopb                              ; vst bmll3, [p3], #64 ; nopxm      // Delay Slot 5
  nopv                            ; nopa                              ; nopb                              ; vst bmlh3, [p3], #64 ; nopxm      // Delay Slot 4
  nopv                            ; nopa                              ; nopb                              ; vst bmhl3, [p3], #64 ; nopxm      // Delay Slot 3
  nopv                            ; nopa                              ; nopb                              ; vst bmhh3, [p3], #64 ; nopxm      // Delay Slot 2
  nopv                            ; nopa                              ; nopb                              ; nops                 ; nopxm      // Delay Slot 1

.Lfunc_end2:
  .size tensor_kernel_64x96x64_bfp16_bfp16_fp32, .Lfunc_end2-tensor_kernel_64x96x64_bfp16_bfp16_fp32

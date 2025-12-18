	.file	"bfp16_emu_mac.cpp"
	.section	.text.bfp16_emu_mac,"ax",@progbits
	.globl	bfp16_emu_mac                       // -- Begin function bfp16_emu_mac
	.p2align	4
	.type	bfp16_emu_mac,@function
bfp16_emu_mac:                              // @bfp16_emu_mac
// %bb.0:                               // %entry
	nopa	;		vldb	 x2, [p1, #0];		nopxm	;		nops	
	vldb	 x4, [p1, #64]
	nop	
	nop	
	nop	
	movxm	r0, #16256
	vlda.conv.fp32.bf16	 cml0, [p0, #0];		vbcst.16	 x0, r0
	mova	r0, #53;		vmov	x1, x0
	mova	r1, #52;		vshuffle	x7, x2, x4, r0
	vlda.conv.fp32.bf16	 cmh0, [p0, #64];		vshuffle	x6, x2, x4, r1
	mova	r0, #60
	vmul.f	dm1, y3, y0, r0
	nop	
	nop	
	vlda	 bmll0, [p2, #0]
	vlda	 bmlh0, [p2, #64]
	vlda	 bmhl0, [p2, #128];		vconv.bfp16ebs8.fp32	 ex0, dm0
	vlda	 bmhh0, [p2, #192];		vconv.bfp16ebs8.fp32	 ex2, dm1
	nop	
	nop	
	mova	r0, #780
	vmac.f	dm0, dm0, ex0, ex2, r0
	nop	
	nop	
	nop	
	nop	
	ret	lr
	vst	 bmll0, [p2, #0]                //  Delay Slot 5
	vst	 bmlh0, [p2, #64]               //  Delay Slot 4
	vst	 bmhl0, [p2, #128]              //  Delay Slot 3
	vst	 bmhh0, [p2, #192]              //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end0:
	.size	bfp16_emu_mac, .Lfunc_end0-bfp16_emu_mac
                                        // -- End function
	.section	".linker-options","e",@llvm_linker_options
	.ident	"clang version 20.0.0 (https://github.com/Xilinx/llvm-aie 0f18d344e89f8c21df668fe389b2e9592e4ab075)"
	.section	".note.GNU-stack","",@progbits
	.addrsig

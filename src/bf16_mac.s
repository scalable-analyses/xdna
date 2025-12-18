	.file	"bf16_mac.cpp"
	.section	.text.bf16_mac,"ax",@progbits
	.globl	bf16_mac                        // -- Begin function bf16_mac
	.p2align	4
	.type	bf16_mac,@function
bf16_mac:                               // @bf16_mac
// %bb.0:                               // %entry
	nopb	;		nopa	;		nops	;		nopx	;		mov	p3, p0;		nopv	
	vldb	wl0, [p0, #0];		mov	p0, p1
	vlda	wl2, [p1, #0];		paddb	[p0], #32;		padds	[p3], #32
	vlda	wh0, [p3, #0];		vldb	wh2, [p0, #0]
	vlda	amhh0, [p2, #32]
	vlda	amhl0, [p2, #0]
	nop	
	nop	
	nop	
	mova	r0, #28
	vmac.f	bmh0, bmh0, x0, x2, r0
	nop	
	nop	
	ret lr	
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	vst	amhh0, [p2, #32]                //  Delay Slot 3
	vst	amhl0, [p2, #0]                 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end0:
	.size	bf16_mac, .Lfunc_end0-bf16_mac
                                        // -- End function
	.section	".linker-options","e",@llvm_linker_options
	.ident	"clang version 20.0.0 (https://github.com/Xilinx/llvm-aie 70ac20b17d30b489a8265a10d1c65a5bf8788c15)"
	.section	".note.GNU-stack","",@progbits
	.addrsig

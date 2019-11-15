stencil: stencil.c
	icc -xHost -simd -O3 -ipo -qopt-report=2 -Wall -std=c11 $^ -o $@


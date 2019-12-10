stencil: stencil.c
	mpiicc -xHost -O3 -ipo -qopt-report=2 -Wall -std=c11 $^ -o $@


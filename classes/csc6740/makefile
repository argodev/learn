PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--smart --standalone
PANDOC_PDF_OPTIONS=-V geometry:margin=1in

# Targets and dependencies
.PHONY: clean bsp

all : bsp


bsp :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o bsp_mpi_regillen42.pdf bsp_mpi_regillen42.md

# list0402 :
# 	gcc -g -Wall -o bin/list0402 list0402.c -lpthread

# list0402cpp :
# 	g++ -g -Wall -o bin/list0402cpp list0402.c -lpthread

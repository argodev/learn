
PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--smart --standalone
PANDOC_PDF_OPTIONS=-V geometry:margin=1in

# Targets and dependencies
.PHONY: clean homework

all : clean homework

clean :
	- $(RM) $(EXPORTED_DOCS)

homework :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Chpt4_T00215814.pdf Chpt4_T00215814.md

list0402 :
	gcc -g -Wall -o bin/list0402 list0402.c -lpthread

list0402cpp :
	g++ -g -Wall -o bin/list0402cpp list0402.c -lpthread

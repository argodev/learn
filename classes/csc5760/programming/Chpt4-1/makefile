MKDIR_P := mkdir -p
OUT_DIR := bin
PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--smart --standalone
PANDOC_PDF_OPTIONS=-V geometry:margin=1in

.PHONY: directories all clean zip report

all: clean no4.13.2 no4.13.14 no4.14.4 zip

directories: $(OUT_DIR)

$(OUT_DIR):
	${MKDIR_P} $(OUT_DIR)

$(OUT_DIR)/program: | directories
	touch $(OUT_DIR)/program

clean:
	rm -rf $(OUT_DIR)


no4.13.2: no4.13.2.c directories
	gcc -std=c11 -g -Wall -fopenmp -o bin/no4.13.2 no4.13.2.c -lpthread

no4.13.14: no4.13.14.c directories
	gcc -std=c11 -Wall -fopenmp -o bin/no4.13.14 no4.13.14.c -lpthread

no4.14.4: no4.14.4.c directories
	gcc -std=c11 -Wall -fopenmp -o bin/no4.14.4 no4.14.4.c -lpthread

report :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o readme.pdf readme.md

zip:
	rm -rf Chpt4-1_T00215814
	rm -f Chpt4-1_T00215814.zip
	mkdir -p Chpt4-1_T00215814
	cp no4.13.2.c Chpt4-1_T00215814
	cp no4.13.14.c Chpt4-1_T00215814
	cp no4.14.4.c Chpt4-1_T00215814
	cp timer.h Chpt4-1_T00215814
	cp readme.md Chpt4-1_T00215814
	cp readme.pdf Chpt4-1_T00215814
	zip -r Chpt4-1_T00215814.zip Chpt4-1_T00215814
	rm -rf Chpt4-1_T00215814

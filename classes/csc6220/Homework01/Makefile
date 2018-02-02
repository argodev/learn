PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--standalone
PANDOC_PDF_OPTIONS=-V geometry:margin=1in

# Targets and dependencies
.PHONY: clean homework

all : clean homework

clean :
	- $(RM) $(EXPORTED_DOCS)

homework :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Hw01_T00215814.pdf Hw01_T00215814.md

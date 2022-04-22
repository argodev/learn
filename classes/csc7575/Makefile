PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--standalone
PANDOC_PDF_OPTIONS=-V geometry:margin=1in

# Targets and dependencies
.PHONY: clean review01 review02 final

all : clean review01 review02 final

clean :
	- $(RM) $(EXPORTED_DOCS)

review01 :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o PowerSubstation_review_T00215814.pdf PowerSubstation_review_T00215814.md

review02 :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o RNPPs_review_T00215814.pdf RNPPs_review_T00215814.md

final :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o FinalExam_T00215814.pdf FinalExam_T00215814.md

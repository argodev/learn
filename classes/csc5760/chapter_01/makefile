# Convert all files in this directory that have a .md suffix
SOURCE_DOCS := $(wildcard *.md)

EXPORTED_DOCS=\
 $(SOURCE_DOCS:.md=.pdf)

RM=/bin/rm

PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--smart --standalone
#PANDOC_HTML_OPTIONS=--to html5
PANDOC_PDF_OPTIONS=
#PANDOC_DOCX_OPTIONS=

# Pattern-matcing rules
#%.html : %.md
#	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_HTML_OPTIONS) -o $@ $<

%.pdf : %.md
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o $@ $<

#%.docx : %.md
#	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_DOCX_OPTIONS) -o $@ $<

# Targets and dependencies
.PHONY: all clean

all : $(EXPORTED_DOCS)

clean :
	- $(RM) $(EXPORTED_DOCS)


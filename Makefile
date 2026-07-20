TECTONIC ?= tectonic
TECTONIC_CACHE ?= /tmp/cute-tutorial-tectonic-cache
TECTONIC_FLAGS ?= -Z shell-escape

SLIDES_MAIN := slides/main.tex
SLIDES_OUTDIR := slides/build
SLIDES_PDF := $(SLIDES_OUTDIR)/main.pdf

.PHONY: all slides clean help

all: slides

slides: $(SLIDES_PDF)

$(SLIDES_PDF): $(SLIDES_MAIN)
	@mkdir -p $(SLIDES_OUTDIR)
	env XDG_CACHE_HOME=$(TECTONIC_CACHE) $(TECTONIC) $(TECTONIC_FLAGS) --outdir $(SLIDES_OUTDIR) $(SLIDES_MAIN)

clean:
	rm -rf $(SLIDES_OUTDIR)

help:
	@echo "Targets:"
	@echo "  make        Build $(SLIDES_PDF)"
	@echo "  make clean  Remove generated slide build files"
	@echo ""
	@echo "Variables:"
	@echo "  TECTONIC=$(TECTONIC)"
	@echo "  TECTONIC_CACHE=$(TECTONIC_CACHE)"
	@echo "  TECTONIC_FLAGS=$(TECTONIC_FLAGS)"

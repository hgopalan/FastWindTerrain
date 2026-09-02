# Minimal Sphinx makefile.
#
# -W turns warnings into errors, which is what makes the build useful: a
# malformed table or a broken cross-reference renders as garbage rather
# than failing, so without -W it would pass silently. --keep-going
# reports every such problem in one run instead of only the first.
#
# It lives in SPHINXOPTS rather than in the CI command so that `make
# html` is the same build locally and in CI.

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# html and linkcheck are deliberately NOT listed phony: GNU make will not
# apply a match-anything pattern rule to a target declared .PHONY, so
# doing so silently disables the catch-all below and every build reports
# "nothing to be done".
.PHONY: help clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

# Catch-all: route anything else (html, linkcheck, ...) to sphinx-build.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

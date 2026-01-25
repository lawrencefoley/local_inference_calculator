# Makefile mínimo para documentação Sphinx
#

# Você pode definir estas variáveis pela linha de comando, e também
# a partir do ambiente para as duas primeiras.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = ../_build/pt_BR

# Coloque primeiro para que "make" sem argumento seja como "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Target genérico: roteia todos os targets desconhecidos para o Sphinx
# usando a opção "make mode".  $(O) é um atalho para $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

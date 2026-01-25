# Makefile principal para documentação multilíngue
# Main makefile for multilingual documentation

# Idiomas disponíveis / Available languages
LANGUAGES = en pt_BR

# Idioma padrão / Default language
LANG ?= en

# Comando Sphinx
SPHINXBUILD   ?= sphinx-build
SPHINXOPTS    ?=

.PHONY: help all clean html $(LANGUAGES)

help:
	@echo "Uso / Usage:"
	@echo "  make html LANG=en        - Build English documentation"
	@echo "  make html LANG=pt_BR     - Build Portuguese documentation"
	@echo "  make all                 - Build all languages"
	@echo "  make clean               - Clean build directories"
	@echo ""
	@echo "Usage / Usage:"
	@echo "  make html LANG=en        - Build English documentation"
	@echo "  make html LANG=pt_BR     - Build Portuguese documentation"
	@echo "  make all                 - Build all languages"
	@echo "  make clean               - Clean build directories"

all: $(LANGUAGES)

$(LANGUAGES):
	@echo "Building $@ documentation / Construindo documentação $@"
	@cd $@ && $(MAKE) html

clean:
	@echo "Cleaning build directories / Limpando diretórios de build"
	@rm -rf _build

html:
	@cd $(LANG) && $(MAKE) html
	@echo ""
	@echo "Build complete! / Build completo!"
	@echo "Open the documentation in your browser:"
	@echo "  file://$(CURDIR)/_build/$(LANG)/html/index.html"

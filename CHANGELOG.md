# Changelog

All notable changes to the Academic Glossary Analysis project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation analysis and updates
- Fixed command references in README to match actual file structure
- Updated hierarchy tool commands to use correct module paths

### Changed
- Pipeline documentation now reflects individual script execution rather than non-existent unified scripts
- Hierarchy tool commands now use `hierarchy.` module prefix instead of `generate_glossary.`

### Fixed
- Documentation command examples now match actual file structure
- Corrected module paths for hierarchy builder, evaluator, and visualizer

## [Previous Work] - 2024

### Added
- Multi-level academic glossary generation pipeline (L0-L3)
- Graph-based deduplication system using NetworkX
- Multiple validation modes (rule-based, web-based, LLM-based)
- Interactive hierarchy visualization with Flask web interface
- Sense disambiguation package for ambiguous term detection
- Support for multiple LLM providers (OpenAI, Gemini)
- Tavily search integration for improved web content mining
- Persistent vector storage with FAISS for embedding caching
- Comprehensive evaluation framework with quality metrics

### Key Features
- **Generation Pipeline**: Four-step process per level (extraction, concept extraction, filtering, verification)
- **Validation System**: Multi-mode validation with relevance scoring
- **Deduplication**: Graph-based approach handling transitive relationships
- **Hierarchy Building**: Parent-child relationship establishment across levels
- **Quality Analysis**: Structural metrics, connectivity analysis, issue detection
- **Web Interface**: Interactive exploration with search and metadata viewing
- **Sense Disambiguation**: Ambiguity detection and splitting proposals

### Technical Stack
- **NLP**: NLTK, spaCy, sentence-transformers, rapidfuzz
- **LLM Integration**: OpenAI, Google Gemini APIs with unified interface
- **Data Processing**: Pandas, Polars, NumPy
- **Web Scraping**: BeautifulSoup, Trafilatura, Playwright
- **Visualization**: NetworkX, Matplotlib, Plotly, Flask
- **Graph Analysis**: NetworkX for relationship modeling
- **Vector Storage**: FAISS for persistent embeddings

### Architecture
- Functional programming approach (no OOP)
- Module-based CLI execution pattern
- Multi-level data processing with validation pipelines
- Provider abstraction for LLM services
- Hierarchical data structure with metadata preservation

---

## Notes

- This project focuses on academic concept extraction and hierarchy building
- Follows functional programming principles throughout
- Designed for processing university-level academic structure (College → Department → Research Area → Conference Topic)
- Supports both automated and manual evaluation workflows
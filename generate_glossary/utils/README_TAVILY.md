# Tavily Integration for Web Content Mining

This document explains how to use the Tavily search engine integration with the web content mining tool for technical glossary generation.

## Overview

The web mining tool now supports using Tavily as an alternative search provider to RapidAPI. Tavily is a search engine optimized for AI assistants and generally provides higher quality search results for technical and academic content.

## Setup

To use Tavily integration, you need to:

1. Get a Tavily API key from [Tavily's website](https://tavily.com)
2. Set the API key as an environment variable:

```bash
# Add to your .env file
TAVILY_API_KEY=tvly-YOUR_API_KEY

# Or set directly in your terminal
export TAVILY_API_KEY=tvly-YOUR_API_KEY
```

## Usage

To use Tavily as your search provider:

```bash
python -m generate_glossary.web_miner_cli \
  -i input_terms.txt \
  -o output_file \
  --search-provider tavily
```

### Important Parameters

- `--search-provider tavily`: Use Tavily as the search engine (default is "rapidapi")
- `--min-score`: Minimum educational score threshold for content verification (default: 2.6)
- `--content-threshold`: Quality threshold for raw content to be processed by LLM (default: 2.0)

## Benefits of Using Tavily

- Higher quality search results, optimized for technical content
- Better content extraction with cleaner results
- More accurate raw content retrieval
- Better filtering of low-quality sources
- Reduced noise in search results

## Comparison with RapidAPI

| Feature | Tavily | RapidAPI |
|---------|--------|----------|
| Result quality | Optimized for technical content | General purpose |
| Content extraction | Included in API | Requires additional processing |
| Query optimization | Better handling of technical terms | Basic handling |
| Content filtering | Built-in domain filtering | Manual filtering |
| API limits | 1,000 free credits/month | Depends on subscription |

## Example

To search for technical terms with Tavily and apply stricter verification:

```bash
python -m generate_glossary.web_miner_cli \
  -i input_terms.txt \
  -o output_file \
  --search-provider tavily \
  --min-score 3.0 \
  --content-threshold 2.5 \
  --log-level INFO
```

## Advanced Configuration

You can customize how Tavily searches by modifying the `search_tavily_for_term` function in `tavily_miner.py`. The default configuration:

1. Searches using multiple query variations:
   - The direct term
   - "What is {term}"
   - "{term} definition"
   - "{term} site:wikipedia.org"

2. Excludes low-quality domains like social media

3. Uses advanced search depth for more comprehensive results

## Troubleshooting

If you encounter any issues:

1. Ensure your API key is correctly set
2. Check your Tavily API credit usage
3. Use the `--log-level DEBUG` flag to see detailed logs
4. Make sure you have the required dependencies installed 
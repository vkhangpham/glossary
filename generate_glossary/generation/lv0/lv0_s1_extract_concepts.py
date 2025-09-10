"""Level 0 Step 1: Extract concepts from college/school names using LLM - Simple wrapper version."""

from generate_glossary.generation.wrapper_utils import create_s1_wrapper

# Create the wrapper with Level 0 configuration
main, test = create_s1_wrapper(level=0)

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        provider = sys.argv[sys.argv.index("--test") + 1] if len(sys.argv) > sys.argv.index("--test") + 1 else "openai"
        test(provider=provider)
    else:
        provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
        main(provider=provider)
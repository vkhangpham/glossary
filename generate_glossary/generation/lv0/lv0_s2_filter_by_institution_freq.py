"""Level 0 Step 2: Filter concepts by institution frequency - Simple wrapper version."""

from generate_glossary.generation.wrapper_utils import create_s2_wrapper

# Create the wrapper with Level 0 configuration
main, test = create_s2_wrapper(level=0)

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test()
    else:
        main()
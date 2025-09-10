"""Level 1 Step 1: Extract department concepts from college data using simplified wrapper approach."""

from ..wrapper_utils import create_s1_wrapper, create_cli_wrapper

# Create wrapper functions
main, test = create_s1_wrapper(level=1)

if __name__ == "__main__":
    create_cli_wrapper(1, "s1", main, test)
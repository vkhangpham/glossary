"""Level 2 Step 1: Extract research area concepts from department data using simplified wrapper approach."""

from ..wrapper_utils import create_s1_wrapper, create_cli_wrapper

# Create wrapper functions
main, test = create_s1_wrapper(level=2)

if __name__ == "__main__":
    create_cli_wrapper(2, "s1", main, test)
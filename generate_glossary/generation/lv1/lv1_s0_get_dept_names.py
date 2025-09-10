#!/usr/bin/env python3
"""Level 1 Step 0: Extract Department Names from College Web Pages using simplified wrapper approach."""

from ..wrapper_utils import create_s0_wrapper, create_cli_wrapper

# Create wrapper functions
main, test = create_s0_wrapper(level=1)

if __name__ == "__main__":
    create_cli_wrapper(1, "s0", main, test)
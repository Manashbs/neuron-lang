"""
Neuron CLI
==========
Command-line interface for the Neuron programming language.

Usage:
    neuron                    Start REPL
    neuron run <file.nl>      Run a Neuron file
    neuron compile <file.nl>  Compile to bytecode
    neuron --version          Show version
    neuron --help             Show help
"""

import sys
import os

# Add parent directory to path so neuron can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron import __version__
from neuron.utils.colors import Colors, NeuronColors


def main():
    args = sys.argv[1:]

    if not args:
        # Start REPL
        from neuron.repl import REPL
        repl = REPL()
        repl.run()
        return

    cmd = args[0]

    if cmd in ("--version", "-v"):
        print(f"Neuron v{__version__}")
        return

    if cmd in ("--help", "-h"):
        print_help()
        return

    if cmd == "run":
        if len(args) < 2:
            print(f"{NeuronColors.ERROR}Error: No file specified{Colors.RESET}")
            print("Usage: neuron run <file.nl>")
            sys.exit(1)
        run_file(args[1])
        return

    if cmd == "compile":
        if len(args) < 2:
            print(f"{NeuronColors.ERROR}Error: No file specified{Colors.RESET}")
            print("Usage: neuron compile <file.nl>")
            sys.exit(1)
        compile_file(args[1])
        return

    # Package manager commands
    if cmd == "get":
        if len(args) < 2:
            print(f"{NeuronColors.ERROR}Error: No package specified{Colors.RESET}")
            print("Usage: neuron get <package>")
            sys.exit(1)
        from neuron.package_manager import install_package
        install_package(args[1], args[2] if len(args) > 2 else None)
        return

    if cmd == "remove":
        if len(args) < 2:
            print(f"{NeuronColors.ERROR}Error: No package specified{Colors.RESET}")
            print("Usage: neuron remove <package>")
            sys.exit(1)
        from neuron.package_manager import remove_package
        remove_package(args[1])
        return

    if cmd == "packages":
        from neuron.package_manager import list_packages
        list_packages()
        return

    if cmd == "search":
        if len(args) < 2:
            print(f"{NeuronColors.ERROR}Error: No query specified{Colors.RESET}")
            print("Usage: neuron search <package>")
            sys.exit(1)
        from neuron.package_manager import search_packages
        search_packages(args[1])
        return

    # If argument is a file, run it directly
    if os.path.isfile(cmd) and cmd.endswith('.nl'):
        run_file(cmd)
        return

    print(f"{NeuronColors.ERROR}Unknown command: {cmd}{Colors.RESET}")
    print("Run 'neuron --help' for usage information.")
    sys.exit(1)


def run_file(filepath):
    """Run a .nl file."""
    if not os.path.exists(filepath):
        print(f"{NeuronColors.ERROR}Error: File not found: {filepath}{Colors.RESET}")
        sys.exit(1)

    from neuron.interpreter import Interpreter
    from neuron.errors import NeuronError

    interpreter = Interpreter()
    try:
        interpreter.execute_file(filepath)
    except NeuronError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"{NeuronColors.ERROR}Internal Error: {e}{Colors.RESET}")
        sys.exit(1)


def compile_file(filepath):
    """Compile a .nl file to bytecode."""
    if not os.path.exists(filepath):
        print(f"{NeuronColors.ERROR}Error: File not found: {filepath}{Colors.RESET}")
        sys.exit(1)

    from neuron.compiler import Compiler
    from neuron.errors import NeuronError

    try:
        compiler = Compiler()
        output = filepath.replace('.nl', '.nlc')
        compiler.compile_file(filepath, output)
        print(f"{NeuronColors.SUCCESS}✓ Compiled: {filepath} → {output}{Colors.RESET}")
    except NeuronError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"{NeuronColors.ERROR}Compilation Error: {e}{Colors.RESET}")
        sys.exit(1)


def print_help():
    print(f"""
{Colors.BOLD}{Colors.BRIGHT_CYAN}Neuron{Colors.RESET} - AI-First Programming Language v{__version__}

{Colors.BOLD}Running Code:{Colors.RESET}
    neuron                    Start interactive REPL
    neuron run <file.nl>      Interpret and run a file
    neuron compile <file.nl>  Compile to bytecode (.nlc)
    neuron <file.nl>          Run a file (shorthand)

{Colors.BOLD}Package Manager:{Colors.RESET}
    neuron get <package>      Install a package
    neuron remove <package>   Uninstall a package
    neuron packages           List installed packages
    neuron search <package>   Search for a package on PyPI

{Colors.BOLD}Info:{Colors.RESET}
    neuron --version          Show version
    neuron --help             Show this help

{Colors.BOLD}Examples:{Colors.RESET}
    neuron                    # Start coding interactively
    neuron run hello.nl       # Run a program
    neuron get pandas         # Install pandas
    neuron get tensorflow     # Install TensorFlow

{Colors.BOLD}Learn more:{Colors.RESET} https://github.com/neuron-lang/neuron
""")


if __name__ == "__main__":
    main()

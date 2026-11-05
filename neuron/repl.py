"""
Neuron REPL
===========
Interactive Read-Eval-Print Loop with syntax highlighting.
"""

import sys
from neuron.interpreter import Interpreter
from neuron.utils.colors import Colors, NeuronColors, colorize, safe_print, safe_text
from neuron.errors import NeuronError
from neuron import __version__


BANNER = f"""{Colors.BOLD}{Colors.BRIGHT_CYAN}
  ╔═╗╔═╗╦ ╦╦═╗╔═╗╔╗╔
  ║╚╗║╣ ║ ║╠╦╝║ ║║║║
  ╚═╝╚═╝╚═╝╩╚═╚═╝╝╚╝{Colors.RESET}
  {Colors.DIM}v{__version__} - AI-First Programming Language{Colors.RESET}
  {Colors.DIM}Type .help for help, .exit to quit{Colors.RESET}
"""


class REPL:
    """Interactive Neuron REPL."""

    def __init__(self):
        self.interpreter = Interpreter()
        self.history = []
        self.multiline_buffer = ""

    def run(self):
        safe_print(BANNER)

        while True:
            try:
                if self.multiline_buffer:
                    prompt = f"{NeuronColors.PROMPT}  ... {Colors.RESET}"
                else:
                    prompt = f"{NeuronColors.PROMPT}  >> {Colors.RESET}"

                line = input(safe_text(prompt, getattr(sys.stdout, "encoding", None) or "utf-8"))

                # Commands
                if not self.multiline_buffer:
                    if line.strip() == ".exit":
                        safe_print(f"\n{Colors.DIM}  Goodbye!{Colors.RESET}\n")
                        break
                    if line.strip() == ".help":
                        self._show_help()
                        continue
                    if line.strip() == ".clear":
                        safe_print("\033[2J\033[H")
                        safe_print(BANNER)
                        continue
                    if line.strip() == ".reset":
                        self.interpreter = Interpreter()
                        safe_print(f"  {NeuronColors.SUCCESS}Environment reset.{Colors.RESET}")
                        continue

                # Multi-line detection
                stripped = line.rstrip()
                if stripped.endswith(':'):
                    self.multiline_buffer += line + "\n"
                    continue
                elif self.multiline_buffer:
                    if line.strip() == "":
                        # Empty line ends multiline input
                        source = self.multiline_buffer
                        self.multiline_buffer = ""
                        self._execute(source)
                    else:
                        self.multiline_buffer += line + "\n"
                    continue

                if not line.strip():
                    continue

                self.history.append(line)
                self._execute(line)

            except EOFError:
                safe_print(f"\n{Colors.DIM}  Goodbye! 🧠{Colors.RESET}\n")
                break
            except KeyboardInterrupt:
                self.multiline_buffer = ""
                safe_print(f"\n  {Colors.DIM}(interrupted){Colors.RESET}")

    def _execute(self, source):
        try:
            result = self.interpreter.execute(source)
            if result is not None:
                display = self.interpreter._to_string(result)
                safe_print(f"  {NeuronColors.SUCCESS}→ {display}{Colors.RESET}")
        except NeuronError as e:
            safe_print(str(e))
        except Exception as e:
            safe_print(f"  {NeuronColors.ERROR}Internal Error: {e}{Colors.RESET}")

    def _show_help(self):
        safe_print(f"""
  {Colors.BOLD}Neuron REPL Commands:{Colors.RESET}
    {NeuronColors.KEYWORD}.help{Colors.RESET}    Show this help
    {NeuronColors.KEYWORD}.exit{Colors.RESET}    Exit the REPL
    {NeuronColors.KEYWORD}.clear{Colors.RESET}   Clear the screen
    {NeuronColors.KEYWORD}.reset{Colors.RESET}   Reset the environment

  {Colors.BOLD}Language Quick Reference:{Colors.RESET}
    {NeuronColors.KEYWORD}do{Colors.RESET} name():           Define a function
    {NeuronColors.KEYWORD}give{Colors.RESET} value            Return a value
    {NeuronColors.KEYWORD}show{Colors.RESET}("text")          Print output
    {NeuronColors.KEYWORD}when{Colors.RESET} cond:            If statement
    {NeuronColors.KEYWORD}each{Colors.RESET} x {NeuronColors.KEYWORD}within{Colors.RESET} list:  For loop
    {NeuronColors.KEYWORD}repeat{Colors.RESET} cond:          While loop
    {NeuronColors.KEYWORD}entity{Colors.RESET} Name:          Define a class
    {NeuronColors.KEYWORD}brain{Colors.RESET} Name:           Define an AI model
    {NeuronColors.KEYWORD}quantum{Colors.RESET} Name:        Define a compile-only quantum circuit
    {NeuronColors.KEYWORD}grab{Colors.RESET} module            Import a library

  Compile quantum blocks with: neuron compile file.nl --target qasm3

  {Colors.BOLD}Available Libraries:{Colors.RESET}
    tensor, neural, optim, data, math,
    io, plot, metrics, preprocess, vision, nlp
""")

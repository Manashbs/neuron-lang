"""
Neuron Package Manager
======================
Install, remove, and list Python packages from within Neuron.

Commands:
    neuron get <package>        Install a package
    neuron remove <package>     Uninstall a package
    neuron packages             List installed packages
    neuron search <query>       Search for packages
"""

import subprocess
import sys
from neuron.utils.colors import Colors, NeuronColors


def get_pip_executable():
    """Get the pip command that matches the current Python."""
    return [sys.executable, "-m", "pip"]


def install_package(name, version=None):
    """Install a package using pip."""
    pkg = f"{name}=={version}" if version else name
    print(f"  {NeuronColors.INFO}Installing {pkg}...{Colors.RESET}")
    print()

    cmd = get_pip_executable() + ["install", pkg, "--quiet", "--no-warn-script-location"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  {NeuronColors.SUCCESS}Done! '{name}' installed successfully.{Colors.RESET}")
        print(f"  {Colors.DIM}Use it in Neuron: grab {name}{Colors.RESET}")
        return True
    else:
        # Try again with verbose output to show error
        cmd = get_pip_executable() + ["install", pkg, "--no-warn-script-location"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  {NeuronColors.SUCCESS}Done! '{name}' installed successfully.{Colors.RESET}")
            return True
        print(f"  {NeuronColors.ERROR}Failed to install '{name}'.{Colors.RESET}")
        if result.stderr:
            # Show last meaningful line of error
            lines = [l for l in result.stderr.strip().split('\n') if l.strip()]
            if lines:
                print(f"  {Colors.DIM}{lines[-1]}{Colors.RESET}")
        return False


def remove_package(name):
    """Uninstall a package."""
    print(f"  {NeuronColors.INFO}Removing {name}...{Colors.RESET}")

    cmd = get_pip_executable() + ["uninstall", name, "-y", "--quiet"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  {NeuronColors.SUCCESS}Done! '{name}' removed.{Colors.RESET}")
        return True
    else:
        print(f"  {NeuronColors.ERROR}Failed to remove '{name}'.{Colors.RESET}")
        return False


def list_packages():
    """List all installed packages."""
    cmd = get_pip_executable() + ["list", "--format=columns"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        print(f"  {Colors.BOLD}Installed Packages:{Colors.RESET}")
        print()
        for line in lines:
            print(f"  {line}")
        print()
        count = max(0, len(lines) - 2)  # subtract header lines
        print(f"  {Colors.DIM}{count} packages installed{Colors.RESET}")
    else:
        print(f"  {NeuronColors.ERROR}Failed to list packages.{Colors.RESET}")


def search_packages(query):
    """Search for packages on PyPI."""
    print(f"  {NeuronColors.INFO}Searching for '{query}'...{Colors.RESET}")
    print()

    # pip search is deprecated, use a simple HTTP request to PyPI
    try:
        import urllib.request
        import json
        url = f"https://pypi.org/pypi/{query}/json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Neuron-Lang/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            info = data.get('info', {})
            print(f"  {Colors.BOLD}{info.get('name', query)}{Colors.RESET} v{info.get('version', '?')}")
            print(f"  {Colors.DIM}{info.get('summary', 'No description')}{Colors.RESET}")
            print()
            print(f"  Install: {NeuronColors.KEYWORD}neuron get {info.get('name', query)}{Colors.RESET}")
    except Exception:
        print(f"  {Colors.DIM}Package '{query}' not found on PyPI.{Colors.RESET}")
        print(f"  {Colors.DIM}Try: neuron get {query}{Colors.RESET}")


def auto_install_and_import(name):
    """Try to import a package; if it fails, auto-install it, then import."""
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        print(f"  {NeuronColors.WARNING}Package '{name}' not found. Installing...{Colors.RESET}")
        if install_package(name):
            import importlib
            try:
                return importlib.import_module(name)
            except ImportError:
                # Package name might differ from import name
                return None
        return None

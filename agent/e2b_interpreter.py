"""
Contains the E2B interpreter for the code executing agent.
"""
from typing import Optional
from e2b_code_interpreter import Sandbox

# Global sandbox instance for session persistence
_sandbox: Optional[Sandbox] = None


def _initialize_sandbox(sbx: Sandbox) -> None:
    """Set up the sandbox environment after creation."""
    sbx.commands.run("mkdir -p /home/user/scripts /home/user/data /home/user/output")


def get_sandbox() -> Sandbox:
    """Get or create the E2B sandbox instance."""
    global _sandbox
    if _sandbox is None:
        _sandbox = Sandbox.create(timeout=3600)
        _initialize_sandbox(_sandbox)
    return _sandbox
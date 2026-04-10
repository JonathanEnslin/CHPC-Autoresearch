#!/usr/bin/env python3
"""Generate PBS job scripts from template and environment variables."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")


def get_env(key: str, default: str = "") -> str:
    """Get environment variable with fallback."""
    return os.getenv(key, default)


def generate_pbs(
    name: str,
    commands: str,
    queue: str = None,
    walltime: str = None,
    ncpus: str = None,
    mem: str = None,
    ngpus: str = None,
    output_dir: str = None,
) -> str:
    """Generate a PBS script from template.

    Args:
        name: Job name
        commands: Shell commands to run
        queue: PBS queue (default from env)
        walltime: Wall time limit (default from env)
        ncpus: Number of CPUs (default from env)
        mem: Memory allocation (default from env)
        ngpus: Number of GPUs (default from env)
        output_dir: Output directory for PBS script

    Returns:
        Path to generated PBS script
    """
    template_path = project_root / "templates" / "experiment.pbs.template"
    if not template_path.exists():
        print(f"Error: Template not found at {template_path}", file=sys.stderr)
        sys.exit(1)

    template = template_path.read_text()

    # Fill in values from args or env defaults
    values = {
        "job_name": name,
        "queue": queue or get_env("AUTORESEARCH_DEFAULT_QUEUE", "gpu_1"),
        "project_id": get_env("CHPC_PROJECT_ID", "CSCI0000"),
        "ncpus": ncpus or get_env("AUTORESEARCH_DEFAULT_NCPUS", "4"),
        "mem": mem or get_env("AUTORESEARCH_DEFAULT_MEM", "64gb"),
        "ngpus": ngpus or get_env("AUTORESEARCH_DEFAULT_NGPUS", "1"),
        "walltime": walltime or get_env("AUTORESEARCH_DEFAULT_WALLTIME", "12:00:00"),
        "lustre_path": get_env("CHPC_LUSTRE_PATH", "/mnt/lustre/users/USERNAME"),
        "repo_name": get_env("CHPC_REPO_NAME", "chpc_autoresearch"),
        "email": get_env("CHPC_EMAIL", "you@example.com"),
        "python_module": get_env("CHPC_MODULE_PYTHON", "chpc/python/anaconda/3-2024.10.1"),
        "commands": commands,
    }

    script = template.format(**values)

    # Write to file
    out_dir = Path(output_dir) if output_dir else project_root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.pbs"
    out_path.write_text(script)

    print(f"Generated PBS script: {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate PBS job scripts")
    parser.add_argument("--name", required=True, help="Job name")
    parser.add_argument("--commands", required=True, help="Commands to run (quoted string)")
    parser.add_argument("--queue", help="PBS queue (default: from .env or gpu_1)")
    parser.add_argument("--walltime", help="Wall time (default: from .env or 12:00:00)")
    parser.add_argument("--ncpus", help="Number of CPUs (default: from .env or 4)")
    parser.add_argument("--mem", help="Memory (default: from .env or 64gb)")
    parser.add_argument("--ngpus", help="Number of GPUs (default: from .env or 1)")
    parser.add_argument("--output-dir", help="Output directory (default: experiments/)")

    args = parser.parse_args()

    generate_pbs(
        name=args.name,
        commands=args.commands,
        queue=args.queue,
        walltime=args.walltime,
        ncpus=args.ncpus,
        mem=args.mem,
        ngpus=args.ngpus,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

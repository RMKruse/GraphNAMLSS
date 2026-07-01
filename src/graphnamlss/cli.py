from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graphnamlss.config import ConfigError, load_experiment_config, validate_experiment_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphnamlss")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run_parser = subparsers.add_parser(
        "dry-run", help="Validate an experiment config and record resolved settings."
    )
    dry_run_parser.add_argument("--config", required=True, help="Path to a YAML config.")

    args = parser.parse_args(argv)

    if args.command == "dry-run":
        return _dry_run(Path(args.config))

    parser.error(f"Unknown command: {args.command}")
    return 2


def _dry_run(config_path: Path) -> int:
    try:
        config = load_experiment_config(config_path)
        validate_experiment_config(config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    resolved = config.to_resolved_dict()
    resolved["dry_run"] = True
    resolved["config_path"] = str(config_path)

    config.output.directory.mkdir(parents=True, exist_ok=True)
    resolved_path = config.output.directory / "resolved_config.json"
    resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")

    print(json.dumps(resolved, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

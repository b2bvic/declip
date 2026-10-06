"""Temporary legacy CLI bridge, replaced by P10."""

from __future__ import annotations

import click

from .contracts import DeclipError


def main() -> None:
    from .legacy.core import cli

    try:
        cli(standalone_mode=False)
    except DeclipError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(exc.exit_code) from exc
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from exc
    except click.Abort as exc:
        click.echo("Aborted!", err=True)
        raise SystemExit(1) from exc

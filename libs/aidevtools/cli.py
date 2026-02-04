"""AI Dev Tools CLI entry point."""

try:
    from prettycli import CLI
    _HAS_PRETTYCLI = True
except ImportError:
    _HAS_PRETTYCLI = False
    CLI = None


def main():
    """Main entry point for aidev CLI."""
    if not _HAS_PRETTYCLI:
        print("Error: prettycli not installed or not in path")
        print("Please ensure libs/prettycli is properly configured")
        return 1

    cli = CLI("aidev")
    cli.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)

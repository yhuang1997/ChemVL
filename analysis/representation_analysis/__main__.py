from analysis.representation_analysis.cli import main

if __name__ == "__main__":
    from utils.notebook_quiet_logging import apply_notebook_quiet_logging

    apply_notebook_quiet_logging()
    raise SystemExit(main())


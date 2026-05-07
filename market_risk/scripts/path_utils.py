from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)

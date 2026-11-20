"""Prepare the pinned CELLO checkout used by the CASAGE experiment driver."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/zjgao02/CELLO.git"
REVISION = "d5c2885073b366c767b1e60198b12efdfeb93c4d"


def git(directory, *args):
    subprocess.run(["git", "-C", str(directory), *args], check=True)


def main():
    destination = ROOT / ".cello"
    if destination.exists():
        raise SystemExit(f"{destination} already exists. Use the existing checkout or move it before preparing a new one.")

    with TemporaryDirectory(prefix=".cello-setup-", dir=ROOT) as temporary:
        checkout = Path(temporary) / "checkout"
        checkout.mkdir()
        git(checkout, "init", "--quiet")
        git(checkout, "remote", "add", "origin", UPSTREAM)
        git(checkout, "fetch", "--quiet", "--depth=1", "origin", REVISION)
        git(checkout, "checkout", "--quiet", "--detach", REVISION)
        git(checkout, "apply", "--check", str(ROOT / "patches/cello-casage.patch"))
        git(checkout, "apply", str(ROOT / "patches/cello-casage.patch"))
        for module in sorted((ROOT / "casage").glob("*.py")):
            shutil.copyfile(module, checkout / "cello" / module.name)
        checkout.rename(destination)

    print(f"CASAGE runtime prepared in {destination}")


if __name__ == "__main__":
    main()

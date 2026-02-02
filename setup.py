from pathlib import Path
from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent

long_description = (BASE_DIR / "README.md").read_text(encoding="utf-8")


def load_requirements():
    req_path = BASE_DIR / "requirements.txt"
    if not req_path.exists():
        return []
    reqs = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        reqs.append(line)
    return reqs


setup(
    name="parc",
    version="0.1.0",
    description="PARC: Physics-based Augmentation with Reinforcement Learning for Character Controllers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mshoe/PARC",
    packages=find_packages(exclude=("tests", "doc")),
    include_package_data=True,
    install_requires=load_requirements(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "run-motionscope=scripts.run_motionscope:main",
        ]
    },
)

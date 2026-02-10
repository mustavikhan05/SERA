import argparse
import contextlib
import docker
import json
import os
import pathlib
import re
import requests
import subprocess
import sys
import traceback

from dataclasses import dataclass, field
from typing import Optional

from swesmith.build_repo.try_install_py import main as try_install_main
from swesmith.constants import LOG_DIR_ENV
from swesmith.profiles import registry
from swesmith.profiles.base import RepoProfile
from swesmith.profiles.python import PythonProfile
from swesmith.profiles.golang import GoProfile
from swesmith.profiles.rust import RustProfile
from swesmith.profiles.javascript import JavaScriptProfile


###############################################################################
# TypeScript profile — extends JavaScriptProfile with TS-specific tooling
###############################################################################

@dataclass
class TypeScriptProfile(JavaScriptProfile):
    """
    SWE-smith profile for TypeScript repositories.

    Inherits from JavaScriptProfile (same Node.js runtime) but adds:
    - TypeScript compilation verification (tsc --noEmit)
    - pnpm as the preferred package manager (with auto-detection fallback)
    - Jelly call-graph extractor for function-level analysis
    - Test framework detection (vitest, jest)

    The Docker image uses node:20-slim as the base and installs pnpm + Jelly
    globally before running the per-repo install script (install_ts.sh).
    """

    # Default base image — can be overridden per-repo
    base_image: str = "node:20-slim"

    # Default install commands for TypeScript repos
    install_cmds: list[str] = field(default_factory=lambda: [
        "npm install -g pnpm",
        "npm install -g @cs-au-dk/jelly",
    ])

    @property
    def _install_script(self) -> pathlib.Path:
        """Path to the TypeScript install script used during image build."""
        return pathlib.Path(__file__).parent / "install_ts.sh"

    def _detect_package_manager(self, repo_path: str | pathlib.Path) -> str:
        """
        Detect the package manager from lockfiles in the repo.
        Returns one of: 'pnpm', 'yarn', 'npm'.
        """
        repo_path = pathlib.Path(repo_path)
        if (repo_path / "pnpm-lock.yaml").exists():
            return "pnpm"
        elif (repo_path / "yarn.lock").exists():
            return "yarn"
        return "npm"

    def _detect_test_framework(self, repo_path: str | pathlib.Path) -> Optional[str]:
        """
        Detect the test framework from package.json.
        Returns one of: 'vitest', 'jest', or None.
        """
        repo_path = pathlib.Path(repo_path)
        pkg_json_path = repo_path / "package.json"
        if not pkg_json_path.exists():
            return None
        try:
            with open(pkg_json_path, "r") as f:
                pkg = json.load(f)
            # Check devDependencies and dependencies
            all_deps = {}
            all_deps.update(pkg.get("dependencies", {}))
            all_deps.update(pkg.get("devDependencies", {}))
            if "vitest" in all_deps:
                return "vitest"
            elif "jest" in all_deps:
                return "jest"
            # Also check scripts for test framework references
            scripts = pkg.get("scripts", {})
            test_script = scripts.get("test", "")
            if "vitest" in test_script:
                return "vitest"
            elif "jest" in test_script:
                return "jest"
        except (json.JSONDecodeError, OSError):
            pass
        return None

    @property
    def dockerfile(self) -> str:
        """Generate Dockerfile for TypeScript repos."""
        return f"""FROM node:20-bullseye
RUN apt update && apt install -y git build-essential python3 python3-pip
RUN python3 -m pip install pipx && python3 -m pipx ensurepath && pipx install swe-rex
ENV PATH="$PATH:/root/.local/bin"
RUN npm install -g pnpm@9
RUN git clone https://github.com/{self.mirror_name} /testbed
WORKDIR /testbed
ENV SHARP_IGNORE_GLOBAL_LIBVIPS=1
ENV npm_config_sharp_libvips_local_prebuilds=0
RUN pnpm install --no-frozen-lockfile --ignore-scripts || pnpm install --no-frozen-lockfile || npm install --ignore-scripts || npm install
"""

    def log_parser(self, log: str) -> dict[str, str]:
        """Parse vitest/jest test output logs into pass/fail per test."""
        from swesmith.profiles.javascript import parse_log_vitest
        return parse_log_vitest(log)

    def _get_install_cmd(self, pkg_manager: str) -> str:
        """Return the appropriate install command for a given package manager."""
        cmds = {
            "pnpm": "pnpm install --frozen-lockfile",
            "yarn": "yarn install --frozen-lockfile",
            "npm": "npm ci",
        }
        return cmds.get(pkg_manager, "npm ci")

    def _get_test_cmd(self, test_framework: Optional[str]) -> Optional[str]:
        """Return the appropriate test command for the detected test framework."""
        if test_framework == "vitest":
            return "npx vitest run --reporter=verbose"
        elif test_framework == "jest":
            return "npx jest --passWithNoTests"
        return None


# Map of language names to their base profile classes
LANGUAGE_PROFILES = {
    "python": PythonProfile,
    "go": GoProfile,
    "golang": GoProfile,
    "rust": RustProfile,
    "javascript": JavaScriptProfile,
    "js": JavaScriptProfile,
    "typescript": TypeScriptProfile,
    "ts": TypeScriptProfile,
}

@contextlib.contextmanager                                                         
def without_pyenv():                                                               
    """
    Temporarily remove pyenv from environment.
    """                               
    old_env = os.environ.copy()                                                    
    try:                                                                           
        path_parts = os.environ.get("PATH", "").split(os.pathsep)                  
        os.environ["PATH"] = os.pathsep.join(p for p in path_parts if "pyenv" not in p.lower())                                                                      
        for var in ["PYENV_VERSION", "PYENV_DIR", "PYENV_ROOT", "PYENV_SHELL"]:    
            os.environ.pop(var, None)                                              
        yield                                                                      
    finally:                                                                       
        os.environ.clear()                                                         
        os.environ.update(old_env)  

def parse_image_ref(image: str):
    """
    Parses docker image name into (namespace, repo, tag).
    """
    image = image.strip()

    # Split tag (last ':' that is not part of a registry host:port).
    # For Docker Hub, most users won't provide a registry host; keep it simple.
    if ":" in image and "/" in image.split(":")[0]:
        # e.g. something/weird:tag (safe to split)
        name, tag = image.rsplit(":", 1)
    elif ":" in image and image.count(":") == 1:
        name, tag = image.rsplit(":", 1)
    else:
        name, tag = image, "latest"

    # If no namespace provided, it's an "official image" under library/
    if "/" not in name:
        namespace, repo = "library", name
    else:
        namespace, repo = name.split("/", 1)

    # Basic sanity
    if not re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", namespace):
        # Docker Hub namespaces are typically lowercase; relax if needed.
        pass

    return namespace, repo, tag


def dockerhub_tag_exists(image: str, timeout=10) -> bool:
    namespace, repo, tag = parse_image_ref(image)
    url = f"https://hub.docker.com/v2/repositories/{namespace}/{repo}/tags/{tag}/"
    r = requests.get(url, timeout=timeout)

    if r.status_code == 200:
        return True
    if r.status_code == 404:
        return False

    # Other statuses can happen (rate limits, auth needed, etc.)
    raise RuntimeError(f"Unexpected status {r.status_code}: {r.text[:300]}")

def create_profile_class(
    owner: str,
    repo: str,
    commit: str,
    language: str,
    install_cmds: Optional[list[str]] = None,
    test_cmd: Optional[str] = None,
    org_dh: Optional[str] = None,
    org_gh: Optional[str] = None,
    python_version: Optional[str] = None,
) -> type[RepoProfile]:
    """
    Create a SWE-smith RepoProfile class for the given repository.
    """
    # Validate language
    language = language.lower()
    if language not in LANGUAGE_PROFILES:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: {', '.join(LANGUAGE_PROFILES.keys())}"
        )
    base_profile = LANGUAGE_PROFILES[language]
    # Create class name: RepoCommit8chars
    commit_short = commit[:8]
    class_name = f"{repo.title()}{commit_short}"

    # Build class attributes and annotations
    attrs = {
        "owner": owner,
        "repo": repo,
        "commit": commit,
    }
    annotations = {
        "owner": str,
        "repo": str,
        "commit": str,
    }

    # Add optional overrides
    if org_dh:
        attrs["org_dh"] = org_dh
        annotations["org_dh"] = str
    if org_gh:
        attrs["org_gh"] = org_gh
        annotations["org_gh"] = str
    if install_cmds:
        attrs["install_cmds"] = field(default_factory=lambda: install_cmds)
        annotations["install_cmds"] = list[str]
    if test_cmd:
        attrs["test_cmd"] = test_cmd
        annotations["test_cmd"] = str
    if python_version and language == "python":
        attrs["python_version"] = field(default=python_version)
        annotations["python_version"] = str

    # Add annotations to attrs
    attrs["__annotations__"] = annotations

    # Create the dataclass
    profile_class = dataclass(
        type(class_name, (base_profile,), attrs)
    )

    return profile_class

def docker_image_exists(image_name: str) -> bool:
    cp = subprocess.run(
        ["docker", "image", "inspect", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cp.returncode == 0

def build_profile_image(
    profile: RepoProfile,
    language: str,
    create_mirror: bool = False,
    push_image: bool = False,
    force: bool = False,
    package_name: str = None
) -> tuple[bool, Optional[str]]:
    """
    Build Docker image for a SWE-smith profile.
    """
    try:
        # Check if image already exists
        if not force:
            client = docker.from_env()
            try:
                client.images.get(profile.image_name)
                print(f"✓ Image already exists: {profile.image_name}")
                return True, None
            except docker.errors.ImageNotFound:
                pass

        print(f"Building profile for: {profile.owner}/{profile.repo}")
        print(f"Commit: {profile.commit}")
        print(f"Image: {profile.image_name}")

        # Step 1: Create GitHub mirror
        if create_mirror:
            print("Step 1/4: Creating GitHub mirror...")
            profile.create_mirror()
            print(f"✓ Mirror created: {profile.mirror_name}")
        else:
            print("Step 1/4: Skipping GitHub mirror creation")

        # Step 2: Generate environment file (language-specific)
        if language == "python":
            print("\nStep 2/4: Generating environment YAML file...")
            print("This may take several minutes...")

            if isinstance(profile, PythonProfile):
                env_yml_path = profile._env_yml
                if os.path.exists(env_yml_path) and not force:
                    print(f"✓ Environment file already exists: {env_yml_path}")
                else:
                    install_script = pathlib.Path("./sera/datagen/data/generate/install.sh")
                    if not install_script.exists():
                        return False, f"Install script not found at {install_script}"
                    # Set python version env var for install.sh
                    if hasattr(profile, 'python_version'):
                        os.environ['SWESMITH_PYTHON_VERSION'] = profile.python_version
                    # Call try_install_py directly
                    try_install_main(
                        repo=f"{profile.owner}/{profile.repo}",
                        install_script=str(install_script),
                        commit=profile.commit,
                        no_cleanup=False,
                        force=force,
                    )
                    print(f"✓ Environment file created: {env_yml_path}")
                    if package_name:
                        with open(env_yml_path, "r") as f:
                            lines = f.readlines()
                        with open(env_yml_path, "w") as f:
                            for line in lines:
                                should_skip = False
                                for pn in package_name:
                                    # print(f"- {pn}==")
                                    # print(line)
                                    if line.strip().startswith(f"- {pn}==") or line.strip().startswith(f"- {pn.lower()}=="):
                                        should_skip = True
                                if not should_skip:
                                    f.write(line)
                        print(f"✓ Filtered package(s) '{package_name}' from environment file")

        elif language in ("typescript", "ts"):
            print("\nStep 2/4: Verifying TypeScript install script exists...")
            install_script = pathlib.Path("./sera/datagen/data/generate/install_ts.sh")
            if not install_script.exists():
                return False, f"TypeScript install script not found at {install_script}"
            print(f"✓ TypeScript install script found: {install_script}")
            # Set environment variables for the install script
            if isinstance(profile, TypeScriptProfile):
                # Pass package manager hint if install_cmds contain one
                for cmd in (profile.install_cmds or []):
                    if "pnpm" in cmd:
                        os.environ['SERA_PKG_MANAGER'] = 'pnpm'
                        break
                    elif "yarn" in cmd:
                        os.environ['SERA_PKG_MANAGER'] = 'yarn'
                        break

        else:
            print("\nStep 2/4: Skipping environment file generation (non-Python repo)")

        # Step 3: Build Docker image
        print("\nStep 3/4: Building Docker image...")
        print("This may take several minutes...")
        profile.build_image()
        print(f"✓ Image built successfully: {profile.image_name}")

        # We override this attribute because its not set correctly by swesmith.
        # So we do our own check with the same logic.
        profile._cache_image_exists = docker_image_exists(profile.image_name)

        # Step 4: Push to Docker Hub (optional)
        if push_image:
            print("\nStep 4/4: Pushing to Docker Hub...")
            profile.push_image()
            print(f"✓ Image pushed: {profile.image_name}")
        else:
            print("\nStep 4/4: Skipping Docker Hub push")

        return True, None

    except Exception as e:
        error_msg = f"Error building {profile.image_name}: {str(e)}"
        traceback.print_exc()
        try:
            build_log_path = LOG_DIR_ENV / profile.repo_name / "build_image.log"
            with open(build_log_path, "r") as f:
                lines = f.readlines()
            print(f"Printing the last 50 lines of the build log from {build_log_path}. Open the full build log for more.")
            for line in lines[-50:]:
                print(line)
        except Exception as e:
            print(f"Could not print debug info from {build_log_path}")
        return False, error_msg

def build_container(
    org_dh: str,
    org_gh: str,
    gh_owner: str,
    repo_name: str,
    commit: str,
    install_cmds: list,
    test_cmd: str = None,
    language: str = "python",
    python_version: str = "3.10",
    package_name: str = None
):
    config = {
        "owner": gh_owner,
        "repo": repo_name,
        "commit": commit,
        "language": language,
        "install_cmds": install_cmds,
        "test_cmd": test_cmd,
        "org_dh": org_dh,
        "org_gh": org_gh,
        "python_version": python_version,
    }

    # python_version is only relevant for Python profiles; remove it for others
    # to avoid passing an unexpected kwarg to create_profile_class
    if language.lower() not in ("python",):
        config.pop("python_version", None)

    try:
        # Create profile class
        with without_pyenv(): # pyenv messes with creating a separate environment
            print("\nCreating profile class...")
            profile_class = create_profile_class(**config)
            print(f"✓ Profile class created: {profile_class.__name__}")

            # Register with registry
            print("Registering with global registry...")
            registry.register_profile(profile_class)
            print(f"✓ Profile registered")

            # Create instance
            profile = profile_class()

            # Check if its already on dockerhub
            if dockerhub_tag_exists(profile.image_name):
                print(f"{profile.image_name} already on dockerhub, returning")
                return profile.image_name

            # Build the image
            success, error = build_profile_image(
                profile,
                language=config['language'],
                create_mirror=True,
                push_image=True if org_dh else False,
                force=True, # TODO: If there's any error in building that user fixes, this allows a retry. Make it a toggle
                package_name=package_name
            )

            if not success:
                print(f"\n❌ Build failed: {error}")
                print(f"Troubleshooting: Make sure that the commit is installable via your installation command")
                return None
                # sys.exit(1)

            print(f"{gh_owner}/{repo_name} at {commit}")
            print(f"\tTest: docker run -it --rm {profile.image_name}")
            print("=" * 60)
            return profile.image_name

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)

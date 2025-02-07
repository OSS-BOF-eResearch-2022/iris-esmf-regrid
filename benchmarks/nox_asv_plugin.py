"""
ASV plug-in providing an alternative ``Environment`` subclass, which uses Nox
for environment management.

"""
from importlib.util import find_spec
from pathlib import Path
from shutil import copy2, copytree
from tempfile import TemporaryDirectory

from nox.sessions import _normalize_path
import yaml

from asv.config import Config
from asv.console import log
from asv.plugins.conda import Conda, _find_conda
from asv.repo import get_repo, Repo
from asv import util as asv_util


# Fetch config variables.
with Path("nox_asv.conf.yaml").open("r") as file:
    config = yaml.load(file, Loader=yaml.Loader)
#: The commit to checkout to first run Nox to set up the environment.
#:  See ``nox_asv.conf.yaml``.
SETUP_COMMIT: str = config["setup_commit"]
#: The path of the noxfile's location relative to the project root.
#:  See ``nox_asv.conf.yaml``.
NOXFILE_REL_PATH: str = config["noxfile_rel_path"]
#: The ``--session`` arg to be used with ``--install-only`` to prep an environment.
#:  See ``nox_asv.conf.yaml``.
SESSION_NAME: str = config["session_name"]


class NoxConda(Conda):
    """
    Manage a Conda environment using Nox, updating environment at each commit.

    Defers environment management to the project's noxfile, which must be able
    to create/update the benchmarking environment using ``nox --install-only``,
    with the ``--session`` specified in :const:`SESSION_NAME` (from
    ``nox_asv_conf.yaml``).

    Notes
    -----
    If not all benchmarked commits support this use of Nox: the plugin will
    need to be modified to prep the environment in other ways.

    """

    tool_name = "nox-conda"

    @classmethod
    def matches(cls, python: str) -> bool:
        """Used by ASV to work out if this type of environment can be used."""
        result = find_spec("nox") is not None
        if result:
            result = super().matches(python)

        if result:
            message = (
                f"NOTE: ASV env match check incomplete. Not possible to know "
                f"if ``nox --session={SESSION_NAME}`` is compatible with "
                f"``--python={python}`` until project is checked out."
            )
            log.warning(message)

        return result

    def __init__(self, conf: Config, python: str, requirements: dict) -> None:
        """
        Parameters
        ----------
        conf: Config instance

        python : str
            Version of Python. Must be of the form "MAJOR.MINOR".

        requirements : dict
            Dictionary mapping a PyPI package name to a version
            identifier string.

        """
        # Need to checkout the project BEFORE the benchmark run - to access a noxfile.
        self.project_temp_checkout = TemporaryDirectory(prefix="nox_asv_checkout_")
        repo = get_repo(conf)
        repo.checkout(self.project_temp_checkout.name, SETUP_COMMIT)
        self.setup_noxfile = Path(self.project_temp_checkout.name) / NOXFILE_REL_PATH

        # Some duplication of parent code - need these attributes BEFORE
        #  running inherited code.
        self._python = python
        self._requirements = requirements
        self._env_dir = conf.env_dir

        # Prepare the actual environment path, to override self._path.
        nox_envdir = str(Path(self._env_dir).absolute() / self.hashname)
        nox_friendly_name = self._get_nox_session_name(python)
        self._nox_path = Path(_normalize_path(nox_envdir, nox_friendly_name))

        # For storing any extra conda requirements from asv.conf.json.
        self._extra_reqs_path = self._nox_path / "asv-extra-reqs.yaml"

        super().__init__(conf, python, requirements)

    @property
    def _path(self) -> str:
        """
        Using a property to override getting and setting in parent classes -
        unable to modify parent classes as this is a plugin.

        """
        return str(self._nox_path)

    @_path.setter
    def _path(self, value) -> None:
        """Enforce overriding of this variable by disabling modification."""
        pass

    def _get_nox_session_name(self, python: str) -> str:
        nox_cmd_substring = (
            f"--noxfile={self.setup_noxfile} "
            f"--session={SESSION_NAME} "
            f"--python={python}"
        )

        list_output = asv_util.check_output(
            ["nox", "--list", *nox_cmd_substring.split(" ")],
            display_error=False,
            dots=False,
        )
        list_output = list_output.split("\n")
        list_matches = list(filter(lambda s: s.startswith("*"), list_output))
        matches_count = len(list_matches)

        if matches_count == 0:
            message = f"No Nox sessions found for: {nox_cmd_substring} ."
            log.error(message)
        elif matches_count > 1:
            message = f"Ambiguous - >1 Nox session found for: {nox_cmd_substring} ."
            log.error(message)
        else:
            line = list_matches[0]
            session_name = line.split(" ")[1]
            return session_name

    def _nox_prep_env(self, setup: bool = False) -> None:
        message = f"Running Nox environment update for: {self.name}"
        log.info(message)

        build_root_path = Path(self._build_root)
        env_path = Path(self._path)

        def copy_asv_files(src_parent: Path, dst_parent: Path) -> None:
            """For copying between self._path and a temporary cache."""
            asv_files = list(src_parent.glob("asv*"))
            # build_root_path.name usually == "project" .
            asv_files += [src_parent / build_root_path.name]
            for src_path in asv_files:
                dst_path = dst_parent / src_path.name
                if not dst_path.exists():
                    # Only cache-ing in case Nox has rebuilt the env @
                    #  self._path. If the dst_path already exists: rebuilding
                    #  hasn't happened. Also a non-issue when copying in the
                    #  reverse direction because the cache dir is temporary.
                    if src_path.is_dir():
                        func = copytree
                    else:
                        func = copy2
                    func(src_path, dst_path)

        with TemporaryDirectory(prefix="nox_asv_cache_") as asv_cache:
            asv_cache_path = Path(asv_cache)
            if setup:
                noxfile_path = self.setup_noxfile
            else:
                # Cache all of ASV's files as Nox may remove and re-build the environment.
                copy_asv_files(env_path, asv_cache_path)
                # Get location of noxfile in cache.
                noxfile_path_build = (
                    build_root_path / self._repo_subdir / NOXFILE_REL_PATH
                )
                noxfile_path = asv_cache_path / noxfile_path_build.relative_to(
                    build_root_path.parent
                )

            nox_cmd = [
                "nox",
                f"--noxfile={noxfile_path}",
                f"--envdir={env_path.parent}",
                f"--session={SESSION_NAME}",
                f"--python={self._python}",
                "--install-only",
                "--no-error-on-external-run",
                "--verbose",
            ]

            _ = asv_util.check_output(nox_cmd)
            if not env_path.is_dir():
                message = f"Expected Nox environment not found: {env_path}"
                log.error(message)

            if not setup:
                # Restore ASV's files from the cache (if necessary).
                copy_asv_files(asv_cache_path, env_path)

        if (not setup) and self._extra_reqs_path.is_file():
            # No need during initial ASV setup - this will be run again before
            #  any benchmarks are run.
            cmd = f"{self.conda} env update -f {self._extra_reqs_path} -p {env_path}"
            asv_util.check_output(cmd.split(" "))

    def _setup(self) -> None:
        """Used for initial environment creation - mimics parent method where possible."""
        try:
            self.conda = _find_conda()
        except IOError as e:
            raise asv_util.UserError(str(e))
        if find_spec("nox") is None:
            raise asv_util.UserError("Module not found: nox")

        message = f"Creating Nox-Conda environment for {self.name} ."
        log.info(message)

        try:
            self._nox_prep_env(setup=True)
        except Exception:
            raise
        finally:
            # No longer need the setup checkout now that the environment has been built.
            self.project_temp_checkout.cleanup()

        # Create an environment.yml file from the requirements in asv.conf.json.
        # No default dependencies to specify - unlike parent - because Nox
        #  includes these by default.
        conda_args, pip_args = self._get_requirements(self.conda)
        if conda_args or pip_args:
            with self._extra_reqs_path.open("w") as req_file:
                req_file.write(f"name: {self.name}\n")
                req_file.write("channels:\n")
                req_file.writelines(
                    [f"   - {channel}\n" for channel in self._conda_channels]
                )
                req_file.write("dependencies:\n")

                # Categorise and write dependencies based on pip vs. conda.
                req_file.writelines([f"   - {package}\n" for package in conda_args])
                if pip_args:
                    # And now specify the packages that are to be installed in the
                    # pip subsection.
                    req_file.write("   - pip:\n")
                    req_file.writelines([f"     - {package}\n" for package in pip_args])

    def checkout_project(self, repo: Repo, commit_hash: str) -> None:
        """Check out the working tree of the project at given commit hash."""
        super().checkout_project(repo, commit_hash)
        self._nox_prep_env()

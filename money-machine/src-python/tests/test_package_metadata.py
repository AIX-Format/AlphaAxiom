"""
Tests for money-machine/package.json metadata.

Coverage:
  - Standard npm metadata fields added in this PR:
    homepage, repository (type/url/directory), bugs (url), license.
  - Removal of the non-standard `aix` field.
  - Field value correctness and format validity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# Resolve the package.json relative to this test file.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PKG_JSON_PATH = _REPO_ROOT / "money-machine" / "package.json"


@pytest.fixture(scope="module")
def pkg() -> dict:
    """Load and parse money-machine/package.json once per test session."""
    assert _PKG_JSON_PATH.exists(), (
        f"package.json not found at {_PKG_JSON_PATH}"
    )
    return json.loads(_PKG_JSON_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# homepage
# ---------------------------------------------------------------------------


def test_homepage_field_is_present(pkg: dict) -> None:
    """PR added a homepage field; it must exist."""
    assert "homepage" in pkg, "package.json is missing the 'homepage' field"


def test_homepage_is_a_non_empty_string(pkg: dict) -> None:
    assert isinstance(pkg["homepage"], str) and pkg["homepage"].strip()


def test_homepage_is_a_valid_https_url(pkg: dict) -> None:
    url = pkg["homepage"]
    assert url.startswith("https://"), (
        f"homepage should use HTTPS, got: {url!r}"
    )


def test_homepage_value(pkg: dict) -> None:
    assert pkg["homepage"] == "https://aqt.axiomid.app"


# ---------------------------------------------------------------------------
# repository
# ---------------------------------------------------------------------------


def test_repository_field_is_present(pkg: dict) -> None:
    """PR added a repository field; it must exist."""
    assert "repository" in pkg, "package.json is missing the 'repository' field"


def test_repository_is_an_object(pkg: dict) -> None:
    assert isinstance(pkg["repository"], dict), (
        "repository must be an object with type/url/directory keys"
    )


def test_repository_has_type(pkg: dict) -> None:
    repo = pkg["repository"]
    assert "type" in repo, "repository must have a 'type' key"
    assert repo["type"] == "git", (
        f"repository.type should be 'git', got: {repo['type']!r}"
    )


def test_repository_has_url(pkg: dict) -> None:
    repo = pkg["repository"]
    assert "url" in repo, "repository must have a 'url' key"
    assert isinstance(repo["url"], str) and repo["url"].strip()


def test_repository_url_is_a_github_git_url(pkg: dict) -> None:
    url = pkg["repository"]["url"]
    assert "github.com" in url, f"repository.url should point to GitHub, got: {url!r}"
    assert url.endswith(".git"), (
        f"repository.url should end with .git, got: {url!r}"
    )


def test_repository_url_value(pkg: dict) -> None:
    assert pkg["repository"]["url"] == "https://github.com/Moeabdelaziz007/AlphaAxiom.git"


def test_repository_has_directory(pkg: dict) -> None:
    repo = pkg["repository"]
    assert "directory" in repo, "repository must have a 'directory' key"
    assert repo["directory"] == "money-machine", (
        f"repository.directory should be 'money-machine', got: {repo['directory']!r}"
    )


# ---------------------------------------------------------------------------
# bugs
# ---------------------------------------------------------------------------


def test_bugs_field_is_present(pkg: dict) -> None:
    """PR added a bugs field; it must exist."""
    assert "bugs" in pkg, "package.json is missing the 'bugs' field"


def test_bugs_is_an_object(pkg: dict) -> None:
    assert isinstance(pkg["bugs"], dict), (
        "bugs must be an object with at least a 'url' key"
    )


def test_bugs_has_url(pkg: dict) -> None:
    bugs = pkg["bugs"]
    assert "url" in bugs, "bugs must have a 'url' key"
    assert isinstance(bugs["url"], str) and bugs["url"].strip()


def test_bugs_url_points_to_issues(pkg: dict) -> None:
    url = pkg["bugs"]["url"]
    assert url.endswith("/issues"), (
        f"bugs.url should point to the GitHub issues page, got: {url!r}"
    )


def test_bugs_url_value(pkg: dict) -> None:
    assert pkg["bugs"]["url"] == "https://github.com/Moeabdelaziz007/AlphaAxiom/issues"


def test_bugs_url_is_https(pkg: dict) -> None:
    url = pkg["bugs"]["url"]
    assert url.startswith("https://"), (
        f"bugs.url should use HTTPS, got: {url!r}"
    )


# ---------------------------------------------------------------------------
# license
# ---------------------------------------------------------------------------


def test_license_field_is_present(pkg: dict) -> None:
    """PR added a license field; it must exist."""
    assert "license" in pkg, "package.json is missing the 'license' field"


def test_license_is_mit(pkg: dict) -> None:
    assert pkg["license"] == "MIT", (
        f"license should be 'MIT', got: {pkg['license']!r}"
    )


# ---------------------------------------------------------------------------
# Removal of non-standard `aix` field
# ---------------------------------------------------------------------------


def test_aix_custom_field_was_removed(pkg: dict) -> None:
    """The non-standard `aix` field was replaced by standard npm metadata
    in this PR. It must no longer appear in package.json.
    """
    assert "aix" not in pkg, (
        "The non-standard 'aix' field should have been removed from package.json"
    )


# ---------------------------------------------------------------------------
# Consistency: repository and bugs belong to the same project
# ---------------------------------------------------------------------------


def test_repository_and_bugs_share_the_same_github_repo(pkg: dict) -> None:
    """repository.url and bugs.url must reference the same GitHub repository."""
    repo_url = pkg["repository"]["url"]
    bugs_url = pkg["bugs"]["url"]
    # Strip .git suffix for comparison.
    repo_base = repo_url.removesuffix(".git")
    assert bugs_url.startswith(repo_base), (
        f"bugs.url ({bugs_url!r}) should be under the same repo as "
        f"repository.url ({repo_url!r})"
    )


# ---------------------------------------------------------------------------
# Mandatory package.json fields are still present
# ---------------------------------------------------------------------------


def test_name_is_still_present(pkg: dict) -> None:
    assert pkg.get("name") == "money-machine"


def test_version_is_still_present(pkg: dict) -> None:
    version = pkg.get("version", "")
    # Semver-ish: major.minor.patch with optional pre-release.
    assert re.match(r"^\d+\.\d+\.\d+", version), (
        f"version should follow semver, got: {version!r}"
    )


def test_private_flag_is_still_set(pkg: dict) -> None:
    """The package should remain private (not published to npm)."""
    assert pkg.get("private") is True

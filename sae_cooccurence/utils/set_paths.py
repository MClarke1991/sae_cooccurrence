import subprocess


def get_git_root():
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
        return git_root
    except subprocess.CalledProcessError:
        print("Not in a git repository")
        return "./"

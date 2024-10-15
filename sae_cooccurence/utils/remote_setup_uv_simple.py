import subprocess
import sys
import os
import shutil
import json

# def get_runpod_env_vars():
#     print("Retrieving RunPod environment variables...")
#     env_vars = {}
#     runpod_vars = [
#         "RUNPOD_SECRET_AWSAccessKey", 
#         "RUNPOD_SECRET_AWSSecretAccess", 
#         "RUNPOD_SECRET_HuggingFaceToken", 
#     ]
    
#     for var in runpod_vars:
#         value = os.environ.get(var)
#         if value:
#             env_vars[var] = value
#         else:
#             print(f"Warning: {var} not found in environment variables.")
    
#     return env_vars

def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error: {e}")
        return False
    return True

def update_and_install_wormhole():
    print("Updating package lists...")
    if not run_command("sudo apt update"):
        if not run_command("apt update"): 
            print("Failed to update package lists")
            return False
    
    print("Installing magic-wormhole...")
    if not run_command("sudo apt install -y magic-wormhole"):
        if not run_command("apt install -y magic-wormhole"):
            print("Failed to install magic-wormhole")
            return False
    
    print("magic-wormhole installed successfully.")
    return True

def install_uv():
    print("Installing uv package manager...")
    run_command("curl -LsSf https://astral.sh/uv/install.sh | sh")
    os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.cargo/bin")

def get_uv_path():
    uv_path = shutil.which("uv")
    if uv_path is None:
        uv_path = os.path.expanduser("~/.cargo/bin/uv")
    return uv_path

def check_python():
    python_path = shutil.which("python")
    if python_path is None:
        python_path = shutil.which("python3")
    if python_path is None:
        raise RuntimeError("No Python interpreter found in system path")
    return python_path

def install_packages(packages):
    uv_path = get_uv_path()
    python_path = check_python()

    for package in packages:
        print(f"Installing {package}...")
        if not run_command(f"{uv_path} pip install --python {python_path} {package}"):
            print(f"Failed to install {package}")
            return False
        print(f"{package} installed successfully.")

    print("Running `uv pip install -e .`...")
    if not run_command(f"{uv_path} pip install --python {python_path} -e ."):
        print("Failed to run `uv pip install -e .`")
        return False
    print("`uv pip install -e .` completed successfully.")

    return True

def setup_git_config():
    email = "matthewaclarke1991@gmail.com"
    name = "MClarke1991"

    print("Setting up git configuration...")
    run_command(f'git config --global user.email "{email}"')
    run_command(f'git config --global user.name "{name}"')
    print("Git configuration completed successfully.")

def install_aws_cli():
    print("Installing AWS CLI...")
    
    # Install unzip
    print("Installing unzip...")
    if not run_command("sudo apt-get install -y unzip"):
        if not run_command("apt-get install -y unzip"):
            print("Failed to install unzip")
            return False
    print("unzip installed successfully.")

    # Download AWS CLI
    if not run_command("curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip'"):
        print("Failed to download AWS CLI")
        return False
    
    # Unzip AWS CLI
    if not run_command("unzip awscliv2.zip"):
        print("Failed to unzip AWS CLI")
        return False
    
    # Install AWS CLI
    if not run_command("./aws/install"):
        if not run_command("./aws/install --bin-dir ~/bin --install-dir ~/.local/aws-cli"):
            print("Failed to install AWS CLI")
            return False
    
    print("AWS CLI installed successfully.")
    return True

def configure_aws_cli(env_vars):
    print("Configuring AWS CLI...")
    aws_access_key_id = env_vars.get("RUNPOD_SECRET_AWSAccessKey") or input("Enter your AWS Access Key ID: ")
    aws_secret_access_key = env_vars.get("RUNPOD_SECRET_AWSSecretAccess") or input("Enter your AWS Secret Access Key: ")
    aws_region = env_vars.get("AWS_DEFAULT_REGION") or input("Enter your default AWS region (e.g., us-west-2): ")

    config_dir = os.path.expanduser("~/.aws")
    os.makedirs(config_dir, exist_ok=True)
    
    with open(os.path.join(config_dir, "credentials"), "w") as f:
        f.write("[default]\n")
        f.write(f"aws_access_key_id = {aws_access_key_id}\n")
        f.write(f"aws_secret_access_key = {aws_secret_access_key}\n")

    with open(os.path.join(config_dir, "config"), "w") as f:
        f.write("[default]\n")
        f.write(f"region = {aws_region}\n")

    print("AWS CLI configured successfully.")



def create_uv_alias():
    print("Creating uv alias...")
    shell = os.environ.get("SHELL", "")

    if "bash" in shell:
        rc_file = os.path.expanduser("~/.bashrc")
    elif "zsh" in shell:
        rc_file = os.path.expanduser("~/.zshrc")
    else:
        print("Unsupported shell. Please add the alias manually to your shell configuration file.")
        return

    alias_line = f'\nalias uv="{get_uv_path()}"'

    with open(rc_file, "a") as f:
        f.write(alias_line)

    print(f"Alias added to {rc_file}. Please restart your terminal or run 'source {rc_file}' to apply changes.")

if __name__ == "__main__":
    packages = [
        "torch",
        "transformer_lens",
        "typing_extensions",
        "sae_lens",
        "community", 
        "toml",
        "kaleido",
        "pyvis", 
        "nvitop", 
        "seaborn", 
        # "runpod", 
        "gdown", 
        # "pytest_snapshot"
        # "pytest_regressions",
        "streamlit",
        "h5py"
    ]

    if update_and_install_wormhole():
            print("magic-wormhole installation successful.")
    else:
        print("magic-wormhole installation failed, continuing with other installations.")

    install_uv()
    if install_packages(packages):
        setup_git_config()
        create_uv_alias()
        
        # runpod_env_vars = get_runpod_env_vars()
        
        # if install_aws_cli():
        #     configure_aws_cli(runpod_env_vars)
        #     print("AWS CLI installed and configured successfully.")
        # else:
        #     print("AWS CLI installation failed.")
        
        print("All tasks completed successfully.")
    else:
        print("Package installation failed.")

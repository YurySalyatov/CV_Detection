import importlib.util
import subprocess
import sys
import argparse

# Dependency mapping: {import_name: pip_package_name}
DEPENDENCIES = {
    'cv2': 'opencv-python',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'ruamel.yaml': 'ruamel.yaml'
    # Add other dependencies as needed
}


def check_dependencies():
    """Check for missing required packages"""
    missing = []
    for import_name, pip_name in DEPENDENCIES.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing.append(pip_name)
    return missing


def install_packages(packages):
    """Install packages using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *packages])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Dependency checker for YOLO detectors')
    parser.add_argument('--auto-install', action='store_true',
                        help='Automatically install missing packages without prompt', default=False)
    args = parser.parse_args()

    missing = check_dependencies()

    if not missing:
        print("\033[92mAll dependencies are satisfied!\033[0m")
        return

    print("\033[93mMissing dependencies detected:\033[0m")
    for pkg in missing:
        print(f" - {pkg}")

    for pkg in missing:
        if args.auto_install:
            print("\033[94mAuto-installing missing packages...\033[0m")
            success = install_packages(pkg)
        else:
            choice = input("\nDo you want to install missing packages? [y/n]: ").lower()
            if choice == 'y':
                print("\033[94mInstalling packages...\033[0m")
                success = install_packages(pkg)
            else:
                print("\033[93mInstallation cancelled\033[0m")
                continue

        if success:
            print("\033[92mInstallation completed!\033[0m")
        else:
            print("\033[91mError occurred during installation!\033[0m")

    end_installation = check_dependencies()
    if end_installation:
        print("\033[All packages are installed! You can use all detectors!\033[0m")
        return

    print("\033[Some packages are not installed! Please, rerun this script!\033[0m")


if __name__ == "__main__":
    main()

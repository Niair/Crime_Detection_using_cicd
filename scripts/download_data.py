import os
import json
import zipfile
import shutil  # ✅ Added missing import
from pathlib import Path
import subprocess
import sys

class UCFCrimeDataDownloader:
    def __init__(self, project_root=None):
        # Initialize the downloader with project structure
        # Args:
        #     project_root: Root directory of the project (if None, uses current directory)
        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)

        # Define directory structure
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.scripts_dir = self.project_root / "scripts"

        # Create directories if they don't exist
        self.create_directories()

    def create_directories(self):
        # Create necessary directories for the project
        directories = [
            self.data_dir,
            self.raw_data_dir, 
            self.processed_data_dir,
            self.scripts_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created/Verified directory: {directory}")

    def setup_kaggle_credentials(self, username, key):
        # Setup Kaggle API credentials
        # Args:
        #     username: Your Kaggle username
        #     key: Your Kaggle API key
        
        # Create .kaggle directory in user home
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)

        # Create kaggle.json file
        credentials = {
            "username": username,
            "key": key
        }

        kaggle_json_path = kaggle_dir / "kaggle.json"
        with open(kaggle_json_path, 'w') as f:
            json.dump(credentials, f)

        # Set appropriate permissions (important for security)
        os.chmod(kaggle_json_path, 0o600)
        print(f"✓ Kaggle credentials saved to {kaggle_json_path}")

    def check_kaggle_installation(self):
        # Check if kaggle package is installed and available
        try:
            # Check if kaggle command is available
            result = subprocess.run(['kaggle', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Kaggle CLI is installed and accessible")
                return True
            else:
                raise Exception("Kaggle command failed")
        except (FileNotFoundError, Exception):
            print("❌ Kaggle CLI not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            print("✓ Kaggle package installed successfully")
            return True

    def download_dataset(self):
        # Download the UCF Crime dataset from Kaggle
        try:
            print("🔄 Starting dataset download...")

            # Download using kaggle API
            subprocess.run([
                "kaggle", "datasets", "download", 
                "-d", "odins0n/ucf-crime-dataset",
                "-p", str(self.raw_data_dir)
            ], check=True)

            print("✓ Dataset downloaded successfully!")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error downloading dataset: {e}")
            print("Please ensure your Kaggle credentials are correct and you've accepted the competition rules.")
            raise

    def extract_dataset(self):
        # Extract the downloaded zip file
        zip_path = self.raw_data_dir / "ucf-crime-dataset.zip"

        if not zip_path.exists():
            # Try to find any zip file in case the name is different
            zip_files = list(self.raw_data_dir.glob("*.zip"))
            if zip_files:
                zip_path = zip_files[0]
                print(f"📁 Found zip file: {zip_path.name}")
            else:
                print(f"❌ Zip file not found in {self.raw_data_dir}")
                return False

        print("🔄 Extracting dataset...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_dir)

            print("✓ Dataset extracted successfully!")

            # Remove the zip file to save space (optional)
            try:
                zip_path.unlink()
                print("✓ Zip file cleaned up")
            except Exception as e:
                print(f"⚠️ Could not remove zip file: {e}")

            return True

        except zipfile.BadZipFile:
            print("❌ Error: The downloaded file is not a valid zip file")
            return False
        except Exception as e:
            print(f"❌ Error extracting dataset: {e}")
            return False

    def organize_data(self):
        # Organize the extracted data into a better structure
        print("🔄 Organizing data structure...")

        # List contents of raw data directory
        contents = list(self.raw_data_dir.iterdir())
        print(f"Raw data contents: {[item.name for item in contents]}")

        # Create organized structure
        organized_dirs = [
            self.raw_data_dir / "videos",
            self.raw_data_dir / "annotations", 
            self.raw_data_dir / "metadata"
        ]

        for dir_path in organized_dirs:
            dir_path.mkdir(exist_ok=True)

        # Move files to appropriate directories based on file extensions
        for item in self.raw_data_dir.iterdir():
            if item.is_file():
                if item.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    shutil.move(str(item), str(self.raw_data_dir / "videos" / item.name))
                elif item.suffix.lower() in ['.txt', '.csv', '.json', '.xml']:
                    shutil.move(str(item), str(self.raw_data_dir / "annotations" / item.name))
                elif item.suffix.lower() in ['.md', '.readme', '.doc', '.pdf']:
                    shutil.move(str(item), str(self.raw_data_dir / "metadata" / item.name))

        print("✓ Data organization completed")

    def create_data_info_file(self):
        # Create a README file with dataset information
        info_content = '''# UCF Crime Dataset Information

                        ## Dataset Overview
                        - **Source**: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
                        - **Download Date**: Auto-generated
                        - **Total Size**: Check the raw/ directory
                        - **Crime Categories**: Multiple categories of criminal activities

                        ## Directory Structure
                        
                        ## Usage Notes
                        - Videos are in various formats (typically .mp4, .avi)
                        - Process videos into frames or features before training
                        - Split data appropriately for train/val/test

                        ## Data Processing Steps
                        1. Extract frames from videos
                        2. Resize frames to consistent dimensions  
                        3. Create train/validation/test splits
                        4. Generate feature vectors if needed

                        '''

        readme_path = self.data_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(info_content)

        print(f"✓ Created data info file: {readme_path}")

def main():
    # Main function to orchestrate the download process
    print("=" * 60)
    print("UCF Crime Dataset Downloader")
    print("=" * 60)

    # Initialize downloader
    downloader = UCFCrimeDataDownloader()

    # Check kaggle installation
    downloader.check_kaggle_installation()

    # Get Kaggle credentials from user
    print("\n📝 Please provide your Kaggle API credentials:")
    print("   You can find these at: https://www.kaggle.com/settings/account")

    username = input("Enter your Kaggle username: ").strip()
    api_key = input("Enter your Kaggle API key: ").strip()

    if not username or not api_key:
        print("❌ Username and API key are required!")
        return

    try:
        # Setup credentials
        downloader.setup_kaggle_credentials(username, api_key)

        # Download dataset
        downloader.download_dataset()

        # Extract dataset
        if downloader.extract_dataset():
            # Organize data
            downloader.organize_data()

            # Create info file
            downloader.create_data_info_file()

            print("\n" + "=" * 60)
            print("✅ SUCCESS! Dataset download and setup completed!")
            print("=" * 60)
            print(f"📁 Raw data location: {downloader.raw_data_dir}")
            print(f"📁 Processed data location: {downloader.processed_data_dir}")
            print("\n🚀 You can now start working with the UCF Crime dataset!")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check your credentials and internet connection.")

if __name__ == "__main__":
    main()
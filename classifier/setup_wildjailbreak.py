import os
import subprocess
import sys
from rich.console import Console
from rich.panel import Panel
import huggingface_hub

def setup_dataset():
    console = Console()
    
    console.print(Panel.fit(
        "[bold green]WildJailbreak Dataset Setup[/bold green]\n\n"
        "This script will help you set up the WildJailbreak dataset.\n"
        "You need to:\n"
        "1. Have a Hugging Face account\n"
        "2. Accept the dataset terms at https://huggingface.co/datasets/allenai/wildjailbreak\n"
        "3. Set up your Hugging Face token",
        title="Setup Instructions"
    ))
    
    # Check if git-lfs is installed
    try:
        subprocess.run(["git-lfs", "--version"], capture_output=True, check=True)
        console.print("[green]✓ git-lfs is installed[/green]")
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]✗ git-lfs is not installed[/red]")
        console.print("Please install git-lfs from https://git-lfs.com")
        return False
        
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Check if dataset is already cloned
    dataset_path = os.path.join("data", "wildjailbreak")
    if os.path.exists(dataset_path):
        console.print("[yellow]Dataset directory exists. Do you want to re-clone it? (y/n)[/yellow]")
        if input().lower() != 'y':
            return True
        # Remove existing directory
        import shutil
        shutil.rmtree(dataset_path)
            
    # Clone the dataset
    console.print("\n[bold]Cloning dataset...[/bold]")
    try:
        # First try to authenticate with huggingface-cli
        try:
            subprocess.run(["huggingface-cli", "login"], check=True)
        except subprocess.CalledProcessError:
            console.print("[yellow]Authentication failed. Please run 'huggingface-cli login' manually first.[/yellow]")
            return False
            
        # Clone the dataset
        subprocess.run([
            "git", "clone",
            "https://huggingface.co/datasets/allenai/wildjailbreak",
            dataset_path
        ], check=True)
        
        # Pull LFS files
        os.chdir(dataset_path)
        subprocess.run(["git", "lfs", "pull"], check=True)
        os.chdir("../..")
        
        console.print("[green]✓ Dataset cloned successfully[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Error cloning dataset: {e}[/red]")
        console.print("\n[bold yellow]Alternative Setup:[/bold yellow]")
        console.print("1. Visit https://huggingface.co/datasets/allenai/wildjailbreak")
        console.print("2. Accept the dataset terms")
        console.print("3. Run: huggingface-cli login")
        console.print("4. Run: huggingface-cli download allenai/wildjailbreak --local-dir data/wildjailbreak")
        return False
        
if __name__ == "__main__":
    console = Console()
    if setup_dataset():
        console.print("\n[bold green]Setup completed successfully![/bold green]")
        console.print("You can now run test_wildjailbreak.py to verify the setup.")
    else:
        console.print("\n[bold red]Setup failed. Please check the errors above.[/bold red]") 
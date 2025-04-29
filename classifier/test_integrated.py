import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from jailbreak_detector import JailbreakDetector, display_results

def main():
    console = Console()
    
    # Initialize the detector
    console.print(Panel.fit(
        "[bold blue]Initializing Jailbreak Detector...[/bold blue]",
        border_style="blue"
    ))
    
    detector = JailbreakDetector()
    
    console.print(Panel.fit(
        "[bold green]System Ready![/bold green]\n"
        "[yellow]Enter 'quit' to exit[/yellow]\n"
        "[cyan]Enter a prompt to analyze:[/cyan]",
        border_style="green"
    ))
    
    while True:
        try:
            # Get user input
            text = Prompt.ask("\nEnter a prompt to analyze")
            
            if text.lower() == 'quit':
                break
                
            if not text:
                continue
            
            # Analyze the prompt
            console.print("\n[bold]Analyzing prompt...[/bold]")
            result = detector.predict(text)
            
            # Display results
            display_results(result, console)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            continue

if __name__ == "__main__":
    main() 
import asyncio
from classifier.jailbreak_detector import JailbreakDetector, display_results
from rich.console import Console

console = Console()

def analyze_prompt(detector: JailbreakDetector, prompt: str):
    """Analyze a single prompt"""
    print("\n" + "="*50)
    print(f"Analyzing prompt: {prompt}")
    
    # Get analysis results
    result = detector.predict(prompt)
    
    # Display results
    display_results(result, console)

async def main():
    """Main function to run the jailbreak detection system"""
    print("Jailbreak Detection System")
    print("="*50)
    print("Enter 'quit' to exit\n")
    
    # Initialize detector
    detector = JailbreakDetector()
    
    while True:
        try:
            # Get user input
            prompt = input("Enter a prompt to analyze: ")
            
            if prompt.lower() == 'quit':
                break
                
            if not prompt:
                continue
            
            # Analyze the prompt
            analyze_prompt(detector, prompt)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main()) 
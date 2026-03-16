"""
Main entry point for the Advanced RAG application.

"""
import sys
from src.config.settings import Settings
from src.core.pipeline import RAGPipeline
from src.utils.logger import setup_logger

logger = setup_logger("main")


def main():
    """Main CLI entry point."""
    # Get query from command line args or prompt user
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print(" Advanced RAG Agent (type 'quit' to exit)")
        print("-" * 50)
        query = input("\nYou: ").strip()
        
        if query.lower() in ["quit", "exit", "q"]:
            print(" Goodbye!")
            return
    
    if not query:
        print("  Please enter a question.")
        return
    
    # Load settings and run pipeline
    try:
        settings = Settings.load()
        pipeline = RAGPipeline(settings=settings)
        
        print("\n Thinking...")
        response = pipeline.run(query)
        
        print(f"\n Agent: {response}")
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n Error: {str(e)}")
        print("Check logs/rag_agent.log for details.")


if __name__ == "__main__":
    main()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List

from src.utils.logger import setup_logger
logger = setup_logger("document_loader")

class DocumentLoader:
    """Load and split documents for vector store ingestion"""

    def __init__(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ):
        """
        Initialize document loader with text splitter.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"DocumentLoader initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")

    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading file: {file_path}")

        # Use appropriate loader based on file extension
                                              #from langchain_community.document_loaders import TextLoader
        loader = PDFPlumberLoader(str(path))  #  loader = TextLoader(str(path), encoding="utf-8") if .txt files
       
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
        
        return documents
    

    def load_directory(self, directory_path: str, glob_pattern: str = "*.pdf") -> List[Document]:
        """
        Load all matching files from a directory.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: File pattern to match (e.g., "*.txt", "*.pdf")
            
        Returns:
            List of Document objects
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        logger.info(f"Loading documents from: {directory_path} (pattern: {glob_pattern})")
        
        loader = DirectoryLoader(
            str(path),
            glob=glob_pattern,
            loader_cls=PDFPlumberLoader, # TextLoader if .txt files
        )
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s) from {directory_path}")
        
        return documents
    

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        logger.info(f"Splitting {len(documents)} document(s)...")
        
        chunks = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} document(s)")
        return chunks
    

    def load_and_split(
        self, 
        source: str, 
        is_directory: bool = False,
        glob_pattern: str = "*.txt"
    ) -> List[Document]:
        """
        Load documents from source and split into chunks.
        
        Args:
            source: File or directory path
            is_directory: Whether source is a directory
            glob_pattern: File pattern (only used if is_directory=True)
            
        Returns:
            List of chunked Document objects
        """
        if is_directory:
            documents = self.load_directory(source, glob_pattern)
        else:
            documents = self.load_file(source)
        
        chunks = self.split_documents(documents)
        return chunks
# src/services/llm_service.py
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from src.config.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger("llm_service")

import os
from dotenv import load_dotenv
load_dotenv()

groq_key = os.getenv("Groq_API_KEY")

class LLMService:
    """
    Service for LLM initialization and generation.
    """

    def __init__(self, settings: Settings):
        """
        Initialize LLM service.
        
        Args:
            settings: Application settings with LLM config
        """

        self.settings = settings
        self._llm: BaseChatModel | None = None

    def get_llm(self) -> BaseChatModel:
        """
        Get or create LLM instance.
        Uses singleton pattern to avoid re-initialization.
        
        Returns:
            LangChain ChatModel object
        """
        if self._llm is None:
            self._llm = self._create_llm()
            logger.info(f"LLM initialized: {self.settings.llm.model}")
        
        return self._llm
    
    def _create_llm(self) -> BaseChatModel:
        """
        Create LLM based on configuration.
        
        Returns:
            LangChain ChatModel object
        """

        provider = self.settings.llm.provider.lower()
        model = self.settings.llm.model
        temperature = self.settings.llm.temperature

        logger.info(f"Creating LLM: provider={provider}, model={model}")
        
        if provider == "groq":
            return ChatGroq(
                api_key=groq_key,
                model=model,
                temperature=temperature,
                max_tokens=self.settings.llm.max_tokens
            )
        
        elif provider == "ollama":
            return ChatOllama(
                model=model,
                temperature=temperature
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer based on query and retrieved context.
        
        Args:
            query: User question
            context: Concatenated retrieved document content
            
        Returns:
            Generated answer string
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant answering questions based ONLY on the provided context.
        
        Context:
        {context}
        
        Question: {query}
        
        Instructions:
        1. Answer the question using ONLY information from the context above.
        2. If the answer cannot be found in the context, respond with: "I cannot find this information in the provided context."
        3. Be concise and accurate.
        4. Do not make up information or use outside knowledge.
        
        Answer:
        """)
        
        chain = prompt | self.get_llm() | StrOutputParser()
        
        logger.info(f"Generating answer for query: '{query[:50]}...'")
        response = chain.invoke({"query": query, "context": context})
        
        logger.debug(f"Generated answer: {response[:100]}...")
        return response
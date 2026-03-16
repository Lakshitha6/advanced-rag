from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from src.config.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger("evaluation_service")


class EvaluationService:
    """
    Service for evaluating answer quality and groundedness.
    Implements the guardrail: "Answer not in the content."
    """

    def __init__(self, settings: Settings, llm: BaseChatModel):
        """
        Initialize evaluation service.
        
        Args:
            settings: Application settings with evaluation config
            llm: LLM instance for evaluation prompts
        """
        self.settings = settings
        self.llm = llm
        
        # Prompt for groundedness check (Natural Language Inference style)
        self.eval_prompt = ChatPromptTemplate.from_template("""
        You are a grader assessing whether an answer is grounded in / supported by the provided context.
        
        Context:
        {context}
        
        Answer:
        {answer}
        
        Question:
        {question}
        
        Task:
        Determine if the answer is fully supported by the context.
        - Return "yes" if the answer can be directly inferred from or stated in the context.
        - Return "no" if the answer contains information not present in the context, or contradicts the context.
        
        Return ONLY "yes" or "no". No explanation.
        """)
        
        self.chain = self.eval_prompt | self.llm | StrOutputParser()
        
        logger.info("EvaluationService initialized")


    def evaluate_grounding(
            self,
            question: str,
            context: str,
            answer: str
    ) -> tuple[bool, str]:
        """
        Evaluate if answer is grounded in context.
        
        Args:
            question: Original user question
            context: Retrieved document content
            answer: Generated answer to evaluate
            
        Returns:
            Tuple of (is_groundeds: bool, reason: str)
        """

        logger.info("Running groundedness evaluation...")
        logger.debug(f"Question: {question[:50]}...")
        logger.debug(f"Answer: {answer[:100]}...")

        try:
            response = self.chain.invoke({
                "question": question,
                "context": context,
                "answer": answer
            }).strip().lower()

            logger.debug(f"Evaluation response: '{response}'")
            
            # Parse response
            is_groundeds = "yes" in response
            
            if is_groundeds:
                logger.info(" Groundedness check PASSED")
                return True, "Answer supported by context"
            else:
                logger.warning(" Groundedness check FAILED")
                logger.debug(f"Context preview: {context[:200]}...")
                return False, "Answer not supported by context"
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Fail safe: if evaluation errors, assume not grounded
            return False, f"Evaluation error: {str(e)}"
        
    
    def should_use_fallback(self, is_groundeds: bool) -> bool:
        """
        Determine if fallback message should be used.
        
        Args:
            is_groundeds: Result from evaluate_grounding()
            
        Returns:
            True if fallback message should be returned
        """
        return not is_groundeds
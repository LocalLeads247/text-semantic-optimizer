class TextOptimizationError(Exception):
    """Base exception for text optimization errors"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class TextTooLongError(TextOptimizationError):
    """Raised when input text exceeds maximum length"""
    pass

class TextTooShortError(TextOptimizationError):
    """Raised when input text is too short"""
    pass

class InvalidOptimizationLevelError(TextOptimizationError):
    """Raised when an invalid optimization level is specified"""
    pass

class ProcessingError(TextOptimizationError):
    """Raised when text processing fails"""
    pass

class UnsupportedLanguageError(TextOptimizationError):
    """Raised when text language is not supported"""
    pass

class InvalidConfigurationError(TextOptimizationError):
    """Raised when configuration is invalid"""
    pass

class ModelNotFoundError(TextOptimizationError):
    """Raised when required NLP model is not found"""
    pass
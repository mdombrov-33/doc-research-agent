class DocResearchError(Exception):
    pass


class DocumentProcessingError(DocResearchError):
    pass


class TextExtractionError(DocumentProcessingError):
    pass


class UnsupportedFileTypeError(TextExtractionError):
    pass


class EmptyDocumentError(DocumentProcessingError):
    pass


class ModelLoadError(DocResearchError):
    pass


class VectorStoreError(DocResearchError):
    pass


class EmbeddingConfigError(VectorStoreError):
    pass


class RetrievalError(DocResearchError):
    pass


class FusionRetrievalError(RetrievalError):
    pass

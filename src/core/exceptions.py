class DocResearchError(Exception):
    pass


class DocumentProcessingError(DocResearchError):
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    pass


class EmptyDocumentError(DocumentProcessingError):
    pass


class ModelLoadError(DocResearchError):
    pass


class EmbeddingConfigError(DocResearchError):
    pass

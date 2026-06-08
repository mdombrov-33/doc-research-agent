from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.utils.logger import logger


def setup_tracing(endpoint: str) -> None:
    resource = Resource(attributes={"service.name": "doc-research-agent"})
    provider = TracerProvider(resource=resource)
    if endpoint:
        # Strip scheme — gRPC exporter expects host:port with insecure=True for http
        grpc_endpoint = endpoint.removeprefix("http://").removeprefix("https://")
        exporter = OTLPSpanExporter(endpoint=grpc_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("tracing_enabled", endpoint=grpc_endpoint)
    else:
        logger.info("tracing_disabled")
    trace.set_tracer_provider(provider)

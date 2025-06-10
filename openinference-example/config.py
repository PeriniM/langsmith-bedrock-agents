import base64
import logging
import os
from phoenix.otel import register
from dotenv import load_dotenv
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Environment settings
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

# Project configuration
def get_project_name():
    """Get project name from environment, allowing for dynamic updates"""
    return os.environ.get("LANGSMITH_PROJECT", "bedrock-agents")

# LangSmith configuration
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_OTLP_ENDPOINT = os.environ.get("LANGSMITH_OTLP_ENDPOINT", "https://api.smith.langchain.com/otel")

def configure_langsmith():
    """Configure for LangSmith"""
    # Set OpenTelemetry environment variables for LangSmith
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = LANGSMITH_OTLP_ENDPOINT
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"x-api-key={LANGSMITH_API_KEY},Langsmith-Project={get_project_name()}"
    
    # Create resource with service information
    resource = Resource.create({
        "service.name": "bedrock-agent-openinference", 
        "service.version": "0.1.0"
    })
    
    # Create the OpenTelemetry exporter
    otlp_exporter = OTLPSpanExporter(timeout=30)
    console_exporter = ConsoleSpanExporter()
    
    # Create and configure the tracer provider
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    provider.add_span_processor(SimpleSpanProcessor(console_exporter))  # Use SimpleSpanProcessor for console exporter
    
    return provider

def create_tracer_provider(provider=None):
    """Create and configure the tracer provider based on selected provider
    
    Args:
        provider (str, optional): The provider to use. 
            Options: "langsmith"
            If None, defaults to "langsmith"
    
    Returns:
        The configured tracer provider
    """
    # Default to langsmith if no provider specified
    if provider is None:
        provider = "langsmith"
    
    if provider == "langsmith":
        logger.info("Using LangSmith for tracing")
        tracer_provider = configure_langsmith()
    else:
        logger.warning(f"Unknown provider: {provider}, defaulting to LangSmith")
        tracer_provider = configure_langsmith()
    
    return tracer_provider
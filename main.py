import boto3
import uuid
import os
from dotenv import load_dotenv

from opentelemetry import trace, metrics
from opentelemetry.trace import SpanKind
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from observability import observe_response
from observability.attributes import SpanAttributes

# Load environment variables from .env file if it exists
load_dotenv(override=True)

def read_secret(secret: str):
    try:
        with open(f"/etc/secrets/{secret}", "r") as f:
            return f.read().rstrip()
    except Exception:
        return os.environ.get(secret.replace("-", "_").upper(), "")


resource = Resource.create(
    {"service.name": "bedrock-agent", "service.version": "0.0.0"}
)

# Get LangSmith API key and project name
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY", "")
project_name = os.environ.get("LANGSMITH_PROJECT", "bedrock-agents")

# Set OpenTelemetry environment variables for LangSmith
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGSMITH_OTLP_ENDPOINT", "https://api.smith.langchain.com/otel")
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"x-api-key={langsmith_api_key},Langsmith-Project={project_name}"

# Create the OpenTelemetry exporter without specifying the endpoint and headers
# They will be picked up from the environment variables
otlp_exporter = OTLPSpanExporter(
    timeout=30,
)
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("bedrock-agents")

if __name__ == "__main__":
    key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    sec = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    region = os.environ.get("AWS_REGION", "eu-north-1")

    client = boto3.client(
        "bedrock-agent-runtime",
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=sec,
    )
    agent_prompt = "Good evening. What time is it?"
    agent_id = os.environ.get("AGENT_ID", "")
    agent_alias_id = os.environ.get("AGENT_ALIAS_ID", "")
    session_id = str(uuid.uuid4())
    with tracer.start_as_current_span(
        name=f"invoke_agent {agent_id}",
        kind=SpanKind.CLIENT,
        attributes={
            # Add LangSmith-specific attributes
            "langsmith.span.kind": "LLM",
            "langsmith.metadata.user_id": session_id,
            SpanAttributes.OPERATION_NAME: "invoke_agent",
            SpanAttributes.SYSTEM: "aws.bedrock",
            SpanAttributes.AGENT_ID: agent_id,
        },
    ) as rootSpan:
        response = client.invoke_agent(
            inputText=agent_prompt,
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            enableTrace=True,
            streamingConfigurations={
                "streamFinalResponse": False,
                "applyGuardrailInterval": 10,
            },
        )
        output = observe_response(response)
        print(output)

    provider.shutdown()
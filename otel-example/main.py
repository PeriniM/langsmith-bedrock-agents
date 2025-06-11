"""
Example usage of Bedrock Agent LangSmith integration with streaming support.
"""
import time
import boto3
import uuid
import json
import os
from core.timer_lib import timer
from core import instrument_agent_invocation, flush_telemetry

@instrument_agent_invocation
def invoke_bedrock_agent(
    inputText: str, agentId: str, agentAliasId: str, sessionId: str, **kwargs
):
    """Invoke a Bedrock Agent with instrumentation for LangSmith."""
    # Create Bedrock client
    bedrock_rt_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=kwargs.get("aws_region", "eu-north-1"),
        aws_access_key_id=kwargs.get("aws_access_key_id"),
        aws_secret_access_key=kwargs.get("aws_secret_access_key"),
    )
    use_streaming = kwargs.get("streaming", False)
    invoke_params = {
        "inputText": inputText,
        "agentId": agentId,
        "agentAliasId": agentAliasId,
        "sessionId": sessionId,
        "enableTrace": True,  # Required for instrumentation
    }

    # Add streaming configurations if needed
    if use_streaming:
        invoke_params["streamingConfigurations"] = {
            "applyGuardrailInterval": 10,
            "streamFinalResponse": use_streaming,
        }
    response = bedrock_rt_client.invoke_agent(**invoke_params)
    return response

def process_streaming_response(stream):
    """Process a streaming response from Bedrock Agent."""
    full_response = ""
    try:
        for event in stream:
            # Convert event to dictionary if it's a botocore Event object
            event_dict = (
                event.to_response_dict()
                if hasattr(event, "to_response_dict")
                else event
            )
            if "chunk" in event_dict:
                chunk_data = event_dict["chunk"]
                if "bytes" in chunk_data:
                    output_bytes = chunk_data["bytes"]
                    # Convert bytes to string if needed
                    if isinstance(output_bytes, bytes):
                        output_text = output_bytes.decode("utf-8")
                    else:
                        output_text = str(output_bytes)
                    full_response += output_text
    except Exception as e:
        print(f"\nError processing stream: {e}")
    return full_response

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv(override=True)

    # Get LangSmith API key and project name
    langsmith_api_key = os.environ.get("LANGSMITH_API_KEY", "")
    project_name = os.environ.get("LANGSMITH_PROJECT", "bedrock-agents")
    langsmith_otlp_endpoint = os.environ.get("LANGSMITH_OTLP_ENDPOINT_TRACES", "https://api.smith.langchain.com/otel/v1/traces")
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    aws_region = os.environ.get("AWS_REGION", "eu-north-1")

    agent_id = os.environ.get("AGENT_ID", "")
    agent_alias_id = os.environ.get("AGENT_ALIAS_ID", "")
    agent_model_id = os.environ.get("AGENT_MODEL_ID", "")
    
    # Set OpenTelemetry environment variables for LangSmith
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = langsmith_otlp_endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"x-api-key={langsmith_api_key},Langsmith-Project={project_name}"    
    
    session_id = f"session-{int(time.time())}"

    # Tags for filtering in LangSmith
    tags = ["bedrock-agent", "example", "development"]
    
    # Generate a custom trace ID
    trace_id = str(uuid.uuid4())
    
    # Prompt
    question = "What is 2+2?"
    streaming = False # Set streaming mode: True for streaming final response, False for non-streaming

    # Prepare the parameters for agent invocation
    invoke_params = {
        "inputText": question,
        "agentId": agent_id,
        "agentAliasId": agent_alias_id,
        "sessionId": session_id,
        "show_traces": True,
        "SAVE_TRACE_LOGS": True,
        "userId": os.environ.get("USER_ID", ""),
        "tags": tags,
        "trace_id": trace_id,
        "streaming": streaming,
        "model_id": agent_model_id,
        "langsmith_api_key": langsmith_api_key,
        "langsmith_project": project_name,
        "langsmith_api_url": langsmith_otlp_endpoint,
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "aws_region": os.environ.get("AWS_REGION", "eu-north-1"),
    }
    
    # Single invocation that works for both streaming and non-streaming
    response = invoke_bedrock_agent(**invoke_params)

    # Handle the response appropriately based on streaming mode
    if isinstance(response, dict) and "error" in response:
        print(f"\nError: {response['error']}")
    elif streaming and isinstance(response, dict) and "completion" in response:
        print("\n🤖 Agent response (streaming):")
        if "extracted_completion" in response:
            print(response["extracted_completion"])
        else:
            process_streaming_response(response["completion"])
    else:
        # Non-streaming response
        print("\n🤖 Agent response:")
        if isinstance(response, dict) and "extracted_completion" in response:
            print(response["extracted_completion"])
        elif (
            isinstance(response, dict) 
            and "completion" in response
            and hasattr(response["completion"], "__iter__")
        ):
            print("Processing completion:")
            full_response = process_streaming_response(response["completion"])
            print(f"\nFull response: {full_response}")
        else:
            print("Raw response:")
            print(f"{response}")

    # Flush telemetry data
    flush_telemetry()
    timer.reset_all()
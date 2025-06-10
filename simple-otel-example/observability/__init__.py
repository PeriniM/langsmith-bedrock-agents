from opentelemetry import trace as otel
from .attributes import SpanAttributes
import json
from datetime import datetime

tracer = otel.get_tracer("bedrock-agents")

# Store active spans by trace_id rather than agent_id
active_spans = {}


def observe_response(response):
    output = ""
    root_span = otel.get_current_span()
    
    # Clear any stale spans
    active_spans.clear()
    
    if "completion" in response:
        _event_stream = response["completion"]
        for event in _event_stream:
            if "chunk" in event:
                output += event["chunk"].get("bytes").decode("utf8")
            if "trace" in event:
                process_trace(event["trace"])

    # Set attributes on the root span
    root_span.set_attribute(SpanAttributes.COMPLETION, output)
    root_span.set_attribute(SpanAttributes.COMPLETION_ROLE, "assistant")
    root_span.set_attribute(SpanAttributes.LC_COMPLETION, output)
    
    # Add LangSmith-specific attributes
    root_span.set_attribute("gen_ai.completion.0.content", output)
    root_span.set_attribute("gen_ai.completion.0.role", "assistant")
    root_span.set_attribute(SpanAttributes.SPAN_KIND, "LLM")
    
    return output


def process_trace(trace):
    print(trace)
    agent_id = trace["agentId"]
    agent_alias_id = trace["agentAliasId"]
    session_id = trace.get("sessionId", "")
    
    # Create a unique run ID for this trace to group related spans
    trace_run_id = f"{agent_id}_{session_id}"
    
    # Add context to the trace data
    if "routingClassifierTrace" in trace["trace"]:
        routing_trace(trace_run_id, agent_id, trace)
    if "orchestrationTrace" in trace["trace"]:
        orchestration_trace(trace_run_id, agent_id, trace)


def routing_trace(trace_run_id, agent_id, trace):
    routing_data = (
        trace["trace"]["routingClassifierTrace"]
        if "trace" in trace and "routingClassifierTrace" in trace["trace"]
        else {}
    )
    
    # Create a unique identifier for this span
    span_id = f"{trace_run_id}_routing_{trace.get('trace', {}).get('routingClassifierTrace', {}).get('traceId', '')}"
    
    # Create a new span if one doesn't exist for this trace part
    if span_id not in active_spans:
        span = tracer.start_span(
            name="agent_routing",
            attributes={
                SpanAttributes.SPAN_KIND: "CHAIN",
                SpanAttributes.OPERATION_NAME: "route_agent",
                SpanAttributes.SYSTEM: "aws.bedrock",
                SpanAttributes.AGENT_ID: agent_id,
                "gen_ai.trace_id": span_id,
                "gen_ai.component": "routing"
            },
        )
        active_spans[span_id] = span
    
    span = active_spans[span_id]
    
    if "modelInvocationInput" in routing_data:
        model_invoke_data = routing_data["modelInvocationInput"]
        handle_model_invoke_input(span, model_invoke_data)
        
    if "modelInvocationOutput" in routing_data:
        model_output = routing_data["modelInvocationOutput"]
        handle_model_invoke_output(span, model_output)
        
    if "observation" in routing_data:
        if "finalResponse" in routing_data["observation"]:
            response_text = routing_data["observation"].get("finalResponse", {}).get("text", "")
            span.set_attribute(SpanAttributes.COMPLETION, response_text)
            span.set_attribute("gen_ai.completion.0.content", response_text)
            span.set_attribute("gen_ai.completion.0.role", "agent")
            
            # Only end the span when we have a final response
            if span_id in active_spans:
                span.end()
                del active_spans[span_id]
                
        if "agentCollaboratorInvocationOutput" in routing_data["observation"]:
            agent_collab_data = routing_data["observation"]["agentCollaboratorInvocationOutput"]
            name = agent_collab_data.get("agentCollaboratorName", "")
            collab_agent_id = extract_agent_id_from_arn(agent_collab_data)
            span.set_attribute("gen_ai.collaborator.agent_id", collab_agent_id)
            span.set_attribute("gen_ai.collaborator.name", name)


def extract_agent_id_from_arn(agent_collab_data):
    arn = agent_collab_data.get("agentCollaboratorAliasArn", "")

    pos = arn.find(":agent-alias/")
    if pos > 0:
        arn = arn[pos:]
        pos = arn.find("/")
        if pos > 0:
            agent_id = arn[pos + 1:]
            return agent_id
    return None


def orchestration_trace(trace_run_id, agent_id, trace):
    orchestration_data = (
        trace["trace"]["orchestrationTrace"]
        if "trace" in trace and "orchestrationTrace" in trace["trace"]
        else {}
    )
    
    # Create a unique identifier for this orchestration span
    trace_id = orchestration_data.get("traceId", "")
    span_id = f"{trace_run_id}_orchestration_{trace_id}"
    
    # Create a new span if one doesn't exist for this trace part
    if span_id not in active_spans:
        span = tracer.start_span(
            name=f"agent_orchestration",
            attributes={
                SpanAttributes.SPAN_KIND: "CHAIN",
                SpanAttributes.OPERATION_NAME: "orchestrate_agent",
                SpanAttributes.SYSTEM: "aws.bedrock",
                SpanAttributes.AGENT_ID: agent_id,
                "gen_ai.trace_id": trace_id,
                "gen_ai.component": "orchestration"
            },
        )
        active_spans[span_id] = span
    
    span = active_spans[span_id]
    
    if "modelInvocationInput" in orchestration_data:
        model_invoke_data = orchestration_data["modelInvocationInput"]
        handle_model_invoke_input(span, model_invoke_data)
        
    if "modelInvocationOutput" in orchestration_data:
        model_output = orchestration_data["modelInvocationOutput"]
        handle_model_invoke_output(span, model_output)
        
    if "invocationInput" in orchestration_data:
        # This is a tool invocation - create a nested span for it
        tool_name = orchestration_data.get("invocationInput", {}).get("actionGroupInvocationInput", {}).get("function", "")
        action_group = orchestration_data.get("invocationInput", {}).get("actionGroupInvocationInput", {}).get("actionGroupName", "")
        
        tool_span_id = f"{span_id}_tool_{tool_name}"
        if tool_span_id not in active_spans:
            tool_span = tracer.start_span(
                name=f"tool_execution_{tool_name}",
                attributes={
                    SpanAttributes.SPAN_KIND: "TOOL",
                    "gen_ai.tool.name": tool_name,
                    "gen_ai.action_group": action_group
                },
            )
            active_spans[tool_span_id] = tool_span
    
    if "observation" in orchestration_data:
        if "actionGroupInvocationOutput" in orchestration_data["observation"]:
            # This is a tool result
            tool_result = orchestration_data["observation"]["actionGroupInvocationOutput"]
            tool_text = tool_result.get("text", "")
            
            # Find the corresponding tool span
            for key, tool_span in list(active_spans.items()):
                if key.startswith(f"{span_id}_tool_"):
                    tool_span.set_attribute("gen_ai.tool.output", tool_text)
                    tool_span.end()
                    del active_spans[key]
                    break
        
        if "finalResponse" in orchestration_data["observation"]:
            response_text = orchestration_data["observation"].get("finalResponse", {}).get("text", "")
            span.set_attribute(SpanAttributes.COMPLETION, response_text)
            span.set_attribute("gen_ai.completion.0.content", response_text)
            span.set_attribute("gen_ai.completion.0.role", "assistant")
            
            # Only end the span when we have a final response
            if span_id in active_spans:
                span.end()
                del active_spans[span_id]


def handle_model_invoke_input(span, model_invoke_data):
    model_name = model_invoke_data.get("foundationModel", "")
    prompt_text = model_invoke_data.get("text", "")
    
    # Create a unique identifier for this LLM span
    trace_id = model_invoke_data.get("traceId", "")
    
    # Set both GenAI and LangChain compatible attributes
    span.set_attribute(SpanAttributes.RESPONSE_MODEL, model_name)
    span.set_attribute(SpanAttributes.PROMPT, prompt_text)
    span.set_attribute(SpanAttributes.LC_PROMPT, prompt_text)
    
    # Add LangSmith-specific attributes with better structured data
    span.set_attribute(SpanAttributes.SPAN_KIND, "LLM")
    span.set_attribute("gen_ai.request.model", model_name)
    
    # Try to parse the JSON prompt for better visualization
    try:
        prompt_json = json.loads(prompt_text)
        if "messages" in prompt_json:
            for i, msg in enumerate(prompt_json.get("messages", [])):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                span.set_attribute(f"gen_ai.prompt.{i}.role", role)
                span.set_attribute(f"gen_ai.prompt.{i}.content", content)
    except:
        # Fall back to simple prompt capture if JSON parsing fails
        span.set_attribute("gen_ai.prompt.0.content", prompt_text)
        span.set_attribute("gen_ai.prompt.0.role", "user")
    
    inference_config = {}
    if "inferenceConfiguration" in model_invoke_data:
        inference_config = model_invoke_data["inferenceConfiguration"]
    
    temp = inference_config.get("temperature", 0)
    top_k = inference_config.get("topK", 0)
    top_p = inference_config.get("topP", 0)
    
    span.set_attribute(SpanAttributes.TEMPERATURE, temp)
    span.set_attribute(SpanAttributes.TOP_K, top_k)
    span.set_attribute(SpanAttributes.TOP_P, top_p)


def handle_model_invoke_output(span, model_data):
    metadata = model_data.get("metadata", {})
    input_tokens = metadata.get("usage", {}).get("inputTokens", 0)
    output_tokens = metadata.get("usage", {}).get("outputTokens", 0)
    total_tokens = input_tokens + output_tokens
    
    # Include both individual and total token counts
    span.set_attribute(SpanAttributes.USAGE_PROMPT_TOKENS, input_tokens)
    span.set_attribute(SpanAttributes.USAGE_COMPLETION_TOKENS, output_tokens)
    span.set_attribute(SpanAttributes.USAGE_TOTAL_TOKENS, total_tokens)
    
    # Handle raw response if available
    if "rawResponse" in model_data and "content" in model_data["rawResponse"]:
        content = model_data["rawResponse"]["content"]
        
        # Try to parse the response JSON for better visualization
        try:
            response_json = json.loads(content)
            if "output" in response_json and "message" in response_json["output"]:
                message_content = response_json["output"]["message"].get("content", [])
                for i, item in enumerate(message_content):
                    if "text" in item and item["text"]:
                        span.set_attribute(f"gen_ai.completion.{i}.content", item["text"])
                        span.set_attribute(f"gen_ai.completion.{i}.role", "assistant")
        except:
            # Fall back to adding the raw response
            span.set_attribute("gen_ai.raw_response", content)
        
    # Add stop reason if available
    stop_reason = None
    if "rawResponse" in model_data and "stopReason" in model_data["rawResponse"]:
        stop_reason = model_data["rawResponse"].get("stopReason")
        span.set_attribute("gen_ai.stop_reason", stop_reason)
from opentelemetry import trace as otel
from .attributes import SpanAttributes

tracer = otel.get_tracer("bedrock-agents")

agent_aliases_to_name = {}


def observe_response(response):
    output = ""
    span = otel.get_current_span()
    if "completion" in response:
        _event_stream = response["completion"]
        for event in _event_stream:
            if "chunk" in event:
                output += event["chunk"].get("bytes").decode("utf8")
            if "trace" in event:
                process_trace(event["trace"])

    span.set_attribute(SpanAttributes.COMPLETION, output)
    span.set_attribute(SpanAttributes.COMPLETION_ROLE, "agent")
    # Add LangSmith-specific attributes
    span.set_attribute("gen_ai.completion.0.content", output)
    span.set_attribute("gen_ai.completion.0.role", "agent")
    return output


def process_trace(trace):
    print(trace)
    agent_id = trace["agentId"]
    agent_alias_id = trace["agentAliasId"]
    if "routingClassifierTrace" in trace["trace"]:
        routing_trace(agent_id, trace)
    if "orchestrationTrace" in trace["trace"]:
        orchestration_trace(agent_id, trace)
    pass


def routing_trace(agent_id, trace):
    routing_span = get_span(agent_id, "agent_routing")
    routing_data = (
        trace["trace"]["routingClassifierTrace"]
        if "trace" in trace and "routingClassifierTrace" in trace["trace"]
        else {}
    )
    if "modelInvocationInput" in routing_data:
        model_invoke_data = routing_data["modelInvocationInput"]
        handle_model_invoke_input(routing_span, model_invoke_data)
    if "modelInvocationOutput" in routing_data:
        model_output = routing_data["modelInvocationOutput"]
        handle_model_invoke_output(routing_span, model_output)
    if "observation" in routing_data:
        if "finalResponse" in routing_data["observation"]:
            response_text = routing_data["observation"].get("finalResponse", {}).get("text", "")
            routing_span.set_attribute(
                SpanAttributes.COMPLETION,
                response_text,
            )
            # Add LangSmith-specific attribute
            routing_span.set_attribute("gen_ai.completion.0.content", response_text)
            routing_span.set_attribute("gen_ai.completion.0.role", "agent")
            routing_span.end()
        if "agentCollaboratorInvocationOutput" in routing_data["observation"]:
            agent_collab_data = routing_data["observation"][
                "agentCollaboratorInvocationOutput"
            ]
            name = agent_collab_data.get("agentCollaboratorName", "")
            ca = extract_agent_id_from_arn(agent_collab_data)
            print(ca)


def extract_agent_id_from_arn(agent_collab_data):
    arn = agent_collab_data.get("agentCollaboratorAliasArn", "")

    pos = arn.find(":agent-alias/")
    if pos > 0:
        arn = arn[pos:]
        pos = arn.find("/")
        if pos > 0:
            agent_id = arn[pos:]
            return agent_id
    return None


def orchestration_trace(agent_id, trace):
    orchestration_span = get_span(agent_id, f"invoke_agent {agent_id}")
    orchestration_data = (
        trace["trace"]["orchestrationTrace"]
        if "trace" in trace and "orchestrationTrace" in trace["trace"]
        else {}
    )
    if "modelInvocationInput" in orchestration_data:
        model_invoke_data = orchestration_data["modelInvocationInput"]
        handle_model_invoke_input(orchestration_span, model_invoke_data)
    if "modelInvocationOutput" in orchestration_data:
        model_output = orchestration_data["modelInvocationOutput"]
        handle_model_invoke_output(orchestration_span, model_output)
    if "rationale" in orchestration_data:
        rationale_text = orchestration_data["rationale"].get("text", "")
        orchestration_span.set_attribute(
            SpanAttributes.COMPLETION + ".0",
            rationale_text,
        )
        # Add LangSmith-specific attribute for rationale
        orchestration_span.set_attribute("gen_ai.reasoning", rationale_text)
    if "observation" in orchestration_data:
        response_text = orchestration_data["observation"].get("finalResponse", {}).get("text", "")
        orchestration_span.set_attribute(
            SpanAttributes.COMPLETION,
            response_text,
        )
        # Add LangSmith-specific attribute for final response
        orchestration_span.set_attribute("gen_ai.completion.0.content", response_text)
        orchestration_span.set_attribute("gen_ai.completion.0.role", "assistant")
        orchestration_span.end()


def handle_model_invoke_input(span, model_invoke_data):
    model_name = model_invoke_data.get("foundationModel")
    prompt_text = model_invoke_data.get("text")
    
    span.set_attribute(
        SpanAttributes.RESPONSE_MODEL, model_name
    )
    span.set_attribute(SpanAttributes.PROMPT, prompt_text)
    
    # Add LangSmith-specific attributes
    span.set_attribute("langsmith.span.kind", "LLM")
    span.set_attribute("gen_ai.request.model", model_name)
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
    
    span.set_attribute(
        SpanAttributes.USAGE_PROMPT_TOKENS,
        input_tokens,
    )
    span.set_attribute(
        SpanAttributes.USAGE_COMPLETION_TOKENS,
        output_tokens,
    )
    
    # Add LangSmith-specific attributes for token usage
    span.set_attribute("gen_ai.usage.prompt_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.completion_tokens", output_tokens)
    span.set_attribute("gen_ai.usage.total_tokens", input_tokens + output_tokens)
    
    pass
    ## TODO: process content to extract chain of thoughts


span_handler = {}


def get_span(agent_id, name):
    if agent_id in span_handler:
        return span_handler.get(agent_id)
    span = tracer.start_span(
        name=name,
        kind=otel.SpanKind.SERVER,
        attributes={
            SpanAttributes.OPERATION_NAME: "invoke_agent",
            SpanAttributes.SYSTEM: "aws.bedrock",
            SpanAttributes.AGENT_ID: agent_id,
            # Add LangSmith-specific attributes
            "langsmith.span.kind": "LLM",
            "gen_ai.system": "aws.bedrock",
        },
    )
    span_handler[agent_id] = span
    return span
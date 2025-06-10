"""
LangSmith OpenTelemetry field constants for proper compatibility.
These constants are derived from the otel_fields_compatible.go file.
"""

# LangSmith specific attributes
LANGSMITH_TRACE_NAME = "langsmith.trace.name"
LANGSMITH_SPAN_KIND = "langsmith.span.kind"
LANGSMITH_METADATA_PREFIX = "langsmith.metadata"
LANGSMITH_RUN_ID = "langsmith.span.id"
LANGSMITH_TRACE_ID = "langsmith.trace.id"
LANGSMITH_DOTTED_ORDER = "langsmith.span.dotted_order"
LANGSMITH_PARENT_RUN_ID = "langsmith.span.parent_id"
LANGSMITH_SESSION_ID = "langsmith.trace.session_id"
LANGSMITH_SESSION_NAME = "langsmith.trace.session_name"
LANGSMITH_TAGS = "langsmith.span.tags"

# LangSmith metadata keys
LANGSMITH_INVOCATION_PARAMS = "invocation_params"
LANGSMITH_METADATA = "metadata"
LANGSMITH_LS_PROVIDER = "ls_provider"
LANGSMITH_LS_MODEL_NAME = "ls_model_name"

# OpenTelemetry IDs
OTEL_SPAN_ID_KEY = "OTEL_SPAN_ID"
OTEL_TRACE_ID_KEY = "OTEL_TRACE_ID"

# Run types
RUN_TYPE_CHAIN = "chain"
RUN_TYPE_LLM = "llm"
RUN_TYPE_TOOL = "tool"
RUN_TYPE_RETRIEVER = "retriever"
RUN_TYPE_EMBEDDING = "embedding"
RUN_TYPE_PROMPT = "prompt"
RUN_TYPE_PARSER = "parser"

# Token usage fields
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

# Input/Output fields
GEN_AI_PROMPT = "gen_ai.prompt"
GEN_AI_COMPLETION = "gen_ai.completion"

# Tool call fields
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

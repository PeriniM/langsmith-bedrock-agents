# LangSmith Bedrock Agents Observer

This project implements observability for AWS Bedrock Agents using OpenTelemetry and LangSmith. It captures detailed telemetry about agent interactions, model calls, and agent responses.

## Prerequisites

- Python 3.8+
- AWS account with Bedrock access
- LangSmith account and API key

## Installation

1. Clone this repository
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The application requires the following environment variables:

### Required Environment Variables

- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` or the corresponding secrets for AWS authentication
- `AGENT_ID` - The ID of your AWS Bedrock Agent
- `AGENT_ALIAS_ID` - The Alias ID of your AWS Bedrock Agent
- `LANGSMITH_API_KEY` - Your LangSmith API key for trace ingestion

### Optional Environment Variables

- `LANGSMITH_PROJECT` - LangSmith project name (default: "bedrock-agents")
- `AWS_REGION` - AWS region for Bedrock (default: "eu-central-1")

You can also set these variables in a `.env` file:

```
AGENT_ID=your-agent-id
AGENT_ALIAS_ID=your-agent-alias-id
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=your-project-name
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

## Running the Application

Execute the main script to invoke your AWS Bedrock agent and send traces to LangSmith:

```bash
python main.py
```

For a custom prompt (instead of the default "Good evening. What can I do in new york?"), modify the agent_prompt variable in main.py.

## Viewing Traces

After execution, you can view the traces in your [LangSmith dashboard](https://smith.langchain.com/).

## Project Structure

- `main.py` - Main application entry point and OpenTelemetry configuration
- `observability/` - Directory containing modules for trace processing
  - `__init__.py` - Core observability functions
  - `attributes.py` - OpenTelemetry attribute constants

## Troubleshooting

- **No traces visible in LangSmith**: Verify your LANGSMITH_API_KEY and check network connectivity
- **AWS API errors**: Confirm your AWS credentials and region settings
- **Missing environment variables**: Ensure all required variables are set

## License

[MIT License](LICENSE)

import logging
from typing import Dict, Any
import datetime

# It's good practice to set up the logger outside the handler
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for processing Bedrock agent requests.
    
    Args:
        event (Dict[str, Any]): The Lambda event containing action details.
        context (Any): The Lambda context object (unused in this function).
    
    Returns:
        Dict[str, Any]: Response containing the action execution results.
    """
    try:
        # Extract details from the event
        action_group = event['actionGroup']
        function_name = event['function']
        message_version = event.get('messageVersion', '1.0')
        parameters = event.get('parameters', [])

        # --- Helper Functions ---
        def get_time() -> str:
            """Returns the current time as a formatted string."""
            return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        def add_two_numbers(a: int, b: int) -> int:
            """Adds two integers."""
            return a + b
        
        # --- Parameter Parsing ---
        # More robustly convert the list of parameters into a simple dictionary
        param_dict = {param['name']: param['value'] for param in parameters}
        
        result_text = f"The function {function_name} is not supported."

        # --- Function Routing ---
        if function_name == "add_two_numbers":
            number_1_str = param_dict.get('number_1')
            number_2_str = param_dict.get('number_2')

            if number_1_str is not None and number_2_str is not None:
                try:
                    # Perform type conversion here, inside the logic
                    number_1 = int(number_1_str)
                    number_2 = int(number_2_str)
                    result = add_two_numbers(number_1, number_2)
                    result_text = f"The result of adding {number_1} and {number_2} is {result}."
                except ValueError:
                    result_text = "Invalid input. Please provide valid integers for both numbers."
            else:
                result_text = "Missing required parameters. Please provide both number_1 and number_2."
        
        elif function_name == "get_time":
            current_time = get_time()
            result_text = f"The current time is {current_time}."

        # --- Response Formatting ---
        # This is the structure Bedrock expects for a successful function call
        response_body = {
            "TEXT": {
                "body": result_text
            }
        }
        
        action_response = {
            'actionGroup': action_group,
            'function': function_name,
            'functionResponse': {
                'responseBody': response_body  # <-- FIX: Correct variable name used here
            }
        }
        
        # Final response wrapper
        api_response = {
            'messageVersion': message_version,
            'response': action_response
        }

        logger.info('Successfully processed action. Response: %s', api_response)
        return api_response

    except Exception as e:
        # Log the full error for debugging
        logger.error('An unexpected error occurred: %s', str(e), exc_info=True)
        # It's crucial to still return a response in the format Bedrock expects,
        # even for an error. Returning an API Gateway style error will fail.
        # This is a generic failure response.
        return {
            'messageVersion': event.get('messageVersion', '1.0'),
            'response': {
                'actionGroup': event.get('actionGroup'),
                'function': event.get('function'),
                'functionResponse': {
                    'responseBody': {
                        "TEXT": {
                            "body": f"An error occurred while processing the request: {str(e)}"
                        }
                    }
                }
            }
        }
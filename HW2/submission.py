def signature():
    """
    Return your first and last name as a string.
    """
    name = "Yifei Chen"
    return name

def api_key():
    """
    Return your api key as a string.
    """
    key = "AIzaSyBM_UTZtlv5WqJh6LtGMKNYjz6_H5px66I"
    return key

def prompt(question: str) -> str:
    # Do not write or delete anything else outside of this function.
    # Do not change the function name or signature.
    # Do not import any additional libraries or modules.
    # Failure to follow these instructions will result in a 0 score on the assignment.
    
    # Instructions:
    # Input: question (str) - a question to be answered by the model
    # Output: prompt (str) - a prompt to be used by the model

    # String formatting
    # You can add as many variables as you want in this function. You can use the variables you
    # declare to pass into your prompt. To do so, you use the format() method.

    # Your job is to create one single prompt that can be used to answer any GSMK8 question.
    # Use the string .format() method to pass in the question and any variables you declare.
    # Please review the format() method in the Python documentation to understand how to use it.
    
    # HINT: Your context can also depend on the question variable that is passed into the function.
    # HINT: Pay attention to how you answer the question. We require your final answer to be in the form of a number.
    # It must be the last thing in your response, and it should be clearly marked with "#### {answer}"
    # The grader will be looking for this delimiter to find your answer. If it cannot find it, it will be considered incorrect.
    # HINT: Regarding the above, you should think about how you can prompt so that the model can produce a 
    # properly delimited answer. You answer should be either a float or an integer, with no other text or symbols.
    # i.e. $990 is not a valid answer, but 990 is.
    #      30 cats is not a valid answer, but 30.0 is.
    #      Ninty-nine luftballoons is not a valid answer, but 99.0 is.
    # You dont have to fill out all the context variables, they just exist if you need it.
    
    operation_protocol = '''
    "You are a highly skilled mathematician. Your task is to solve the question step by step, "
        "showing all intermediate calculations clearly and ensuring accuracy. "
        "Provide the final result as a number, clearly marked with '#### answer: <answer>', where <answer> is the final result. "
        "Here are some examples to guide you:\n"
        "1. What is 29312620553 + 46767577934? Answer: 76080198487\n"
        "2. What is 37355508 - 99339904? Answer: -61984396\n"
        "3. What is 8679 * 131? Answer: 1136949\n"
        "4. What is 3105 * 838? Answer: 2601990\n"
        "5. What is 1524 * 869? Answer: 1324356\n"
        "Follow these examples when solving the question. Recheck your calculations to ensure accuracy before outputting the final answer."
'''
    format_rules = '''
    Calculation Steps: 
1. Clearly outline every step of the calculation.
2. Ensure that intermediate results are accurate and match the mathematical operations described.
3. Verify the final result before providing the answer. Recheck the result by reversing the operation (e.g., subtraction to verify addition, division to verify multiplication).
"""
'''
    examples = '''Take a deep breath and get accurate answer'''

    logical_analysis = '''
Steps to Analyze the Question: 
1. Parse the numbers and the operation (e.g., addition, subtraction, multiplication) from the question.
2. Confirm the operation type is valid and relevant to the question.
3. Use accurate arithmetic operations to calculate the result.
4. Cross-check the calculation using reverse operations.
5. Provide the answer clearly in the required format.
'''
    prompt = """

        <operation_protocol>
        {operation_protocol}
        </operation_protocol>

        <format_rules>
        {format_rules}
        </format_rules>

        <logical_analysis>
        {logical_analysis}
        </logical_analysis>

        <examples>
        {examples}
        </examples>
        
        <question>
        {question}
        </question>

    """
    
    prompt = prompt.format(question=question, operation_protocol=operation_protocol, format_rules=format_rules, logical_analysis=logical_analysis, examples=examples)
    
    return prompt


# DO NOT MODIFY THIS FUNCTION
# This is a naive prompt. It exists for you to see how you can use the format() method.
def naive_prompt(question: str) -> str:
    
    prompt = '{question} #### answer: <answer>'
    return prompt.format(question=question)

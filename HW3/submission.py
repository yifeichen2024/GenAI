import pandas as pd

OPERATIONS = {'add': '+', 'subtract': '-', 'multiply': '*'}

def prepare_data(path):
    '''
    In this function, you are to read in the train.csv file by passing its path to the function. 
    
    It should return a pandas dataframe with two columns: Input_Text and Solution. Please follow 
    the exact format as shown in the example of one entry below:
    
    Input_Text: 'What is 651475 * 4458?'
    Solution: 'use_calculator(651475, 4458, "multiply")'
    
    That is, you begin with "What is" followed by the two numbers and the operation, and end with a question mark.
    The solution is a string that begins with "use_calculator" followed by the two numbers and the operation in parentheses.
    NOTE: PAY ATTENTION TO THE QUOTES AROUND THE OPERATION.
    NOTE: BE SURE TO FORMAT IT AS num1 */+/- num2 AND NOT num1 multiply/add/subtract num2
    
    Both columns should be of type string. Use f-strings to format the input and solution.
    This function should be able to handle any number of rows in the csv file.
    
    NOTE: The OPERATIONS dictionary is provided to you and can be used to map the operation to the correct symbol.
    
    You only need to implement the for loop.
    '''
    df = pd.read_csv(path)
    df_cleaned = {'Input_Text': [], 'Solution': []}
    
    for k, r in df.iterrows():
        pass # TODO: Implement this for loop 
    
    df_cleaned = pd.DataFrame(df_cleaned)
    return df_cleaned

def use_calculator(num1, num2, operation):
    '''
    This function is the calculator tool. 
    
    It should support addition, subtraction, and multiplication. These operations 
    are passed as strings to the function as the `operation` argument.
    
    The numbers should be converted to integers before the calculations are performed.
    
    The function should return the result of the operation.
    
    Consider using if statements to check the operation and perform the correct calculation.
    '''
    raise NotImplementedError # TODO: implement this function

def api_key():
    # This function should return your Hugging Face API key.
    # this will allow the autograder to pull your model from the hub.
    # if it cannot pull your model from the hub, your assignment will be graded as 0.
    key = None # TODO: implement this function
    return key

def hub_model_name():
    # This function should return the name of the model push to the Hugging Face Hub.
    # it should be in the format of 'username/modelname'
    # if the autograder cannot pull your model from the hub, your assignment will be graded as 0.
    model_name = None # TODO: implement this function
    return model_name

# NO NEED TO MODIFY THIS FUNCTION
def capture_output(output):
    '''
    This function is provided to you and should not be modified.
    
    It captures the output of the language model and makes calculations.
    
    NOTE: We will use this function to extract the calculator function from your model output
    during grading. This should not affect how to train your model.
    '''
    if 'use_calculator(' in output:
        p1 = output.split('use_calculator(')[1].split(',')[0].strip()
        p2 = output.split('use_calculator(')[1].split(',')[1].strip()
        operation = output.split('use_calculator(')[1].split(',')[2].split(')')[0].strip()
        operation = operation.replace('"', '')  
        return use_calculator(int(p1), int(p2), operation)
    return None







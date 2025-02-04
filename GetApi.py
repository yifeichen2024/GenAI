import google.generativeai as genai

genai.configure(api_key="AIzaSyBM_UTZtlv5WqJh6LtGMKNYjz6_H5px66I")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("What is your name?")
print(response.text)
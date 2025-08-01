import google.generativeai as genai
genai.configure(api_key="AIzaSyCUnF5Y3iE1O8VMr1UUyJ50Zdwt1N7DDZ4")
print([m.name for m in genai.list_models()])
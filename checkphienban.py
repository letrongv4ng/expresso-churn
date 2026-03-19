import google.generativeai as genai

genai.configure(api_key="AIzaSyBfASucOYLLXW9yiPyTuu6mH25oF2KcJWA")

# Để check xem máy mày đang "thấy" được những con model nào, chạy cái này:
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name) # Nó sẽ hiện list kiểu 'models/gemini-1.5-flash'

# Sau đó dùng con model mày tìm thấy:
model = genai.GenerativeModel('gemini-1.5-flash') # Hoặc con 3.1 nếu nó hiện trong list
response = model.generate_content("Prompt của mày ở đây")
print(response.text)
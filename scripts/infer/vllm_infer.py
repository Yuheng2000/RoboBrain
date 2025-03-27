from openai import OpenAI
import base64

openai_api_key = "robobrain-123123" 
openai_api_base = "http://172.16.20.50:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

image_path = "./test.png"

with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
base64_img = f"data:image;base64,{encoded_image_text}"

response = client.chat.completions.create(
    model="robobrain",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_img
                    },
                },
                {"type": "text", "text": "Describe the image."},
            ],
        },
    ]
)

content = response.choices[0].message.content
print(content)
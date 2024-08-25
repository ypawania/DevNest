from dotenv import load_dotenv
import openai
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT = "You are an animal expert. You will receive the name of an animal, and you will respond with the following information in order: \
                 The IUCN conservation status, risk assessment to humans (no danger, some danger, very dangerous), what they usually eat and finish off with a fun fact on the animal. \
                 The response should be in the following format: \
                 - **IUCN Conservation Status:** \n \
                 - **Risk Assessment to Humans:** \n \
                 - **Diet:** \n \
                 - **Fun Fact:**"

client = openai.OpenAI(api_key=API_KEY)


def get_response(animal):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": animal},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(get_response("coyote")) # test the function
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-1106:learninggpt:sfbu-bot:9CVU8Zib",
  messages=[{"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Can you provide some data on scholarship availabilities at SFBU?"}]
)
print(completion.choices[0].message.content)

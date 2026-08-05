import os
from openai import OpenAI
import hgls.persona as persona

client = OpenAI(
    api_key=os.environ.get('TOGETHER_API_KEY', ''),
    base_url='https://api.together.xyz/v1'
)

tests = [
    {
        'label': 'Test 1: No system prompt',
        'messages': [
            {'role': 'user', 'content': 'Little Deepak is a good 5-year-old Indian child. Someone said to him: "do you love your amma?" Write only what Deepak should say in reply.'}
        ]
    },
    {
        'label': 'Test 2: With system prompt',
        'messages': [
            {'role': 'system', 'content': persona.PARENT_SYSTEM_PROMPT},
            {'role': 'user', 'content': 'Little Deepak is a good 5-year-old Indian child. Someone said to him: "do you love your amma?" Write only what Deepak should say in reply.'}
        ]
    },
    {
        'label': 'Test 3: Simple question',
        'messages': [
            {'role': 'user', 'content': 'Say hello in one sentence.'}
        ]
    },
    {
        'label': 'Test 4: Direct reply request',
        'messages': [
            {'role': 'user', 'content': 'Someone asked a child: "do you love your mother?" The child replies:'}
        ]
    },
]

for test in tests:
    print(f"\n{test['label']}")
    print('-' * 40)
    try:
        resp = client.chat.completions.create(
            model='openai/gpt-oss-20b',
            max_tokens=80,
            messages=test['messages']
        )
        result = resp.choices[0].message.content
        print(f"Response: {repr(result)}")
        print(f"Finish reason: {resp.choices[0].finish_reason}")
    except Exception as e:
        print(f"Error: {e}")
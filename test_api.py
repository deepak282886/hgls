"""
API diagnostic — shows full response structure.
python test_api.py
"""
import os, json, urllib.request

API_KEY = os.environ.get('TOGETHER_API_KEY', '')
print(f'Key found: {bool(API_KEY)}')

payload = json.dumps({
    'model':       'openai/gpt-oss-20b',
    'messages':    [{'role': 'user', 'content': 'Say the word OK and nothing else.'}],
    'max_tokens':  20,
    'temperature': 0.1,
}).encode()

headers = {
    'Content-Type':  'application/json',
    'Authorization': f'Bearer {API_KEY}',
    'User-Agent':    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

try:
    req = urllib.request.Request(
        'https://api.together.xyz/v1/chat/completions',
        data=payload, headers=headers, method='POST'
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    print('\nFull response:')
    print(json.dumps(data, indent=2))

    # Try every possible content location
    choice = data['choices'][0]
    msg    = choice.get('message', {})
    print(f'\nchoice keys:          {list(choice.keys())}')
    print(f'message keys:         {list(msg.keys())}')
    print(f'content:              {repr(msg.get("content"))}')
    print(f'reasoning_content:    {repr(msg.get("reasoning_content"))}')
    print(f'tool_calls:           {repr(msg.get("tool_calls"))}')
    print(f'finish_reason:        {repr(choice.get("finish_reason"))}')

except Exception as e:
    print(f'Error: {e}')
import json, random
random.seed(20261110)
words = ["amber","basil","cobalt","delta","ember","fjord","garnet","harbor","indigo","juniper","kestrel","lumen","maple","nectar","onyx","pewter","quartz","raven","saffron","tundra"]
def haystack(n, ask):
    lines = [f"Record {i}: the access code for locker {i} is {random.choice(words)}-{random.randint(100,999)}." for i in range(1, n+1)]
    return "Below is a registry of locker access codes.\n" + "\n".join(lines) + f"\n\nQuestion: What is the access code for locker {ask}? Answer: The access code for locker {ask} is"
P = [
 {"name": "sky", "prompt": "Explain in two sentences why the sky appears blue during the day.", "max_tokens": 64},
 {"name": "france", "prompt": "The capital of France is", "max_tokens": 32},
 {"name": "fib", "prompt": "Write a Python function that returns the nth Fibonacci number.\n", "max_tokens": 64},
 {"name": "primes", "prompt": "List three prime numbers greater than 100, separated by commas:", "max_tokens": 32},
 {"name": "long8k", "prompt": haystack(410, 287), "max_tokens": 24},
 {"name": "long30k", "prompt": haystack(1540, 1177), "max_tokens": 24},
]
json.dump(P, open("prompts.json", "w"))
print([ (p["name"], len(p["prompt"])) for p in P])

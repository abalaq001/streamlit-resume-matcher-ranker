import requests
import json
import time

api_url = "https://jsearch.p.rapidapi.com/search"

headers = {
    "X-RapidAPI-Key": "2219efb88bmsh7edef8ab6bd5be3p182abajsnf9b4f3e84893",
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

# List of job queries you want
job_queries = [
    "data scientist in India",
    "machine learning intern in India",
    "AI researcher in India",
    "NLP intern in India",
    "deep learning engineer in India"
]

all_jobs = []

for job in job_queries:
    print(f"🔍 Searching for: {job}")
    querystring = {
        "query": job,
        "page": "1",
        "num_pages": "1"
    }

    response = requests.get(api_url, headers=headers, params=querystring)
    
    if response.status_code == 200:
        results = response.json().get("data", [])
        print(f"✅ Found {len(results)} jobs for: {job}")
        all_jobs.extend(results)
    else:
        print(f"❌ Failed to fetch for '{job}': Status {response.status_code}")
    
    time.sleep(1)  # Pause to avoid hitting rate limits

# Save all collected jobs to a file
with open("job_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(all_jobs, f, ensure_ascii=False, indent=4)

print(f"\n🎉 Done! Collected and saved {len(all_jobs)} job listings.")

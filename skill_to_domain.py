# skill_to_domain.py

from domain_skill_mapping import domain_skills
from collections import defaultdict

# Reverse the mapping: skill -> [domains]
skill_to_domains = defaultdict(set)
for domain, skills in domain_skills.items():
    for skill in skills:
        skill_to_domains[skill.lower()].add(domain)

def infer_domain_from_skills(extracted_skills):
    votes = defaultdict(int)
    for skill in extracted_skills:
        for domain in skill_to_domains.get(skill.lower(), []):
            votes[domain] += 1
    if not votes:
        return "Unknown"
    return max(votes, key=votes.get)

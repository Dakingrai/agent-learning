from dotenv import load_dotenv
load_dotenv()
from langsmith import Client
client = Client()
projects = list(client.list_projects(limit=5))
print('Projects:', [p.name for p in projects])
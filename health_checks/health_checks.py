# Test Service Connectivity
import requests
import subprocess
import json

services_to_test = {
    "FastAPI": "http://localhost:8000/api/v1/health",
    "PostgreSQL (via API)": "http://localhost:8000/api/v1/health", 
    "OpenSearch": "http://localhost:9200/_cluster/health",
    "Airflow": "http://localhost:8080/health"  
}

print("WEEK 2 PREREQUISITE CHECK")
print("=" * 50)

all_healthy = True

for service_name, url in services_to_test.items():
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ {service_name}: Healthy")
        else:
            print(f"✗ {service_name}: HTTP {response.status_code}")
            all_healthy = False
    except requests.exceptions.ConnectionError:
        print(f"✗ {service_name}: Not accessible")
        all_healthy = False
    except Exception as e:
        print(f"✗ {service_name}: {type(e).__name__}")
        all_healthy = False

print()
if all_healthy:
    print("All services healthy! Ready for Week 2 development.")
else:
    print("Some services need attention. Check Week 1 notebook.")
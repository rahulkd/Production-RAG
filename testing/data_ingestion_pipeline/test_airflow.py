# Test Airflow DAGs
import subprocess

print("Test 7: Airflow DAG Status")
print("=" * 40)

print("  Airflow UI Access:")
print("   URL: http://localhost:8080")
print("   Username: admin")
print("   Password: admin")
print()

# Check DAG status using docker exec
try:
    result = subprocess.run(
        ["docker", "exec", "rag-airflow", "airflow", "dags", "list"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        dag_lines = [line for line in lines if 'arxiv' in line.lower() or 'hello' in line.lower()]
        
        print("Available DAGs:")
        for line in dag_lines:
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 3:
                    dag_id = parts[0]
                    is_paused = parts[2]
                    status = "Active" if is_paused == "False" else "Paused"
                    print(f"   - {dag_id}: {status}")
        
        # Check for import errors
        error_result = subprocess.run(
            ["docker", "exec", "rag-airflow", "airflow", "dags", "list-import-errors"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if "docling" in error_result.stderr:
            print("\nKnown Issue: Docling not installed in Airflow container")
            print("   - This is expected for Week 2")
            print("   - DAG structure is complete, runtime needs container fix")
            print("   - Solution: Add docling to Airflow container startup")
        elif error_result.returncode == 0:
            print("\n✓ No DAG import errors found")
        
    else:
        print(f"✗ Could not list DAGs: {result.stderr}")
        
except Exception as e:
    print(f"✗ Airflow test error: {e}")

print("\n  To view DAGs graphically:")
print("   1. Open http://localhost:8080 in your browser")
print("   2. Login with admin/admin")
print("   3. Click on 'arxiv_paper_ingestion' DAG to see the workflow")
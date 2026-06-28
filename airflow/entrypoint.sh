#!/bin/bash
set -e

# Clean up any existing PID files and processes
echo "Cleaning up any existing Airflow processes..."
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true
rm -f /opt/airflow/airflow-webserver.pid
rm -f /opt/airflow/airflow-scheduler.pid

# Wait a moment for processes to fully terminate
sleep 2

# Initialize Airflow database
echo "Initializing Airflow database..."
airflow db init

# Create admin user with admin/admin credentials
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "Admin user already exists"

# Start webserver and scheduler
# Run the webserver in the background WITHOUT --daemon: --daemon writes a pidfile
# and double-forks, which (combined with backgrounding and gunicorn's master)
# trips the "webserver is already running under PID N" check and kills the UI.
# Plain background keeps it as a child process logging to stdout, no pidfile.
echo "Starting Airflow webserver and scheduler..."
airflow webserver --port 8080 &
airflow scheduler

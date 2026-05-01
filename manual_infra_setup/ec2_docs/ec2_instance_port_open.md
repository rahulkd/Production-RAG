To check a service running on "localhost" within an Amazon EC2 instance from your local web browser, you have two primary methods: using SSH Port Forwarding (recommended for security) or Public Access via the instance's IP. [1, 2, 3, 4] 
## Method 1: SSH Port Forwarding (Secure & Direct)
This method creates a secure tunnel between your computer and the EC2 instance, allowing you to access the EC2's localhost as if it were running on your own machine. [5, 6] 

   1. Open your terminal on your local machine.
   2. Run the SSH tunnel command:
   ssh -i "your-key.pem" -L 8080:localhost:4000 ec2-user@your-ec2-public-ip
   * 8080: The port you will use in your local browser.
      * 4000: The port your app is running on inside the EC2 instance. [3, 7, 8, 9] 
   3. Access in Browser: Open http://localhost:8080 on your local computer. [5, 10, 11] 

## Method 2: Public Access via IP
If you want to access the instance directly using its Public IP address, you must configure AWS security settings. [3, 12] 

   1. Configure Security Group:
   * Go to the Amazon EC2 Console.
      * Select your instance, click the Security tab, and select the active Security Group.
      * Edit Inbound Rules: Add a "Custom TCP" rule for your application's port (e.g., 4200 or 8080) and set the Source to "Anywhere-IPv4" (0.0.0.0/0) or your specific IP. [3, 13, 14, 15, 16] 
   2. Update Application Binding:
   * Ensure your application is listening on 0.0.0.0 (all interfaces) rather than just 127.0.0.1 (localhost). If it only listens on localhost, external requests will be refused. [17, 18, 19, 20, 21] 
   3. Access in Browser: Enter http://<EC2-Public-IP>:<Port> (e.g., http://54.12.34.56:4200). [3, 22] 

## Troubleshooting Tips

* Internal Firewall: If you still cannot connect, ensure the instance's internal firewall (like ufw on Ubuntu) allows the port: sudo ufw allow <port>/tcp.
* Verify Listening Port: Run netstat -nltp on the EC2 instance to confirm the service is actually running and see which IP it is bound to.
* Public IP: Use the [EC2 Instance Summary](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html) to verify your Public IPv4 address hasn't changed after a restart. [3, 16, 23, 24, 25] 


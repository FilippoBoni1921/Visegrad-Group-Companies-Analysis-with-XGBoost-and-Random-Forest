# Use an official Python 3.11-slim image as the base image
FROM python:3.11-slim

# Install Python 3.11.4 manually
RUN apt-get update && apt-get install -y python3.11

# Set Python 3.11.4 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --set python3 /usr/bin/python3.11

# Check Python version
RUN python3 --version

# Set the working directory 
WORKDIR .

# Copy the current directory contents into the container 
COPY . .

# Install gcc and python3-dev
RUN apt-get update && apt-get install -y gcc python3-dev

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run py when the container launches
CMD ["python3","./src/main.py"]
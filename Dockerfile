# Use an official PyTorch image as the base image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy all contents to the container
COPY . /app

# Install git
RUN apt-get update && apt-get install -y git

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

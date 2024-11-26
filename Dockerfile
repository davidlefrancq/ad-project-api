FROM python:3.11-slim

# RUN apt-get update && apt-get install -y libgomp1

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./src /app
COPY ./requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "main.py"]

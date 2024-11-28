FROM python:3.11-alpine

# Update the package list
RUN apk update

RUN apk add --no-cache \
    # gcc \
    # musl-dev \
    # python3-dev \
    # libffi-dev \
    # openssl-dev \
    build-base

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./src /app/src
COPY ./pyproject.toml /app/
COPY ./pdm.lock /app/

# Install any needed packages specified in requirements.txt
RUN pip install pdm
RUN pdm install --prod

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["pdm", "run", "start"]

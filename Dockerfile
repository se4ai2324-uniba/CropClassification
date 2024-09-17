FROM python:3.12.0-slim
# Set the working directory in the container

WORKDIR /app
# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Expose
EXPOSE 8000


# Copy the content of the local src directory to the working directory
COPY . .

#Specify the command to run Fastapi
CMD ["uvicorn", "apis.main:app", "--host", "0.0.0.0", "--port", "8000"]
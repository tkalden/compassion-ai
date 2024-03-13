# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10.9

EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

#setting time out to 0 to avoid gunicorn timeout error and let google cloud auto scaling handle the scaling

CMD exec gunicorn --bind :5000 main:app --workers 1 --threads 8 --timeout 0

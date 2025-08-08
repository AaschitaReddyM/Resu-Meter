#using an official python runtime as a parent image
FROM python:3.11-slim

#setting the working directory in the container
WORKDIR /app

#copying the requirements file into the container at /app
COPY requirements.txt

#Installing any required packages specified in the reqts document
RUN pip install --no-cache-dir -r requirements.txt

#now copying the rest of the app code into the container at /app
COPY . .

#making the port at 8501 available to the outside traffic outside this container
EXPOSE 8501

#defining the command to run the application
HEALTHCHECK CMD streamlit hello
CMD ["streamlit", "run", "app.py"]

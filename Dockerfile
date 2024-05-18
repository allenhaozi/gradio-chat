FROM allenhaozi/gradio:4.31.4-py3.10 
WORKDIR /app
COPY chat/app.py /app/app.py

CMD [ "python", "app.py" ]

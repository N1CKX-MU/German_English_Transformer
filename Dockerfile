# Use official PyTorch CPU image
FROM pytorch/pytorch:2.2.2-cpu

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download spaCy language models
RUN python -m spacy download de_core_news_sm
RUN python -m spacy download en_core_web_sm

# Default command
CMD ["python"]
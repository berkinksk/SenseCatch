# Container for the SenseCatch app, serving every model option over gunicorn
# on port 7860.
FROM python:3.12-slim

# System runtime that scikit-learn, scipy, and torch need.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# The platform runs the container as this non-root user.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    NLTK_DATA=/home/user/nltk_data \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

RUN pip install --no-cache-dir --upgrade pip

# CPU build of torch first, from the CPU wheel index.
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user:user requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

# Fetch the NLTK resources the app uses, so the first request does not download them.
RUN python -m nltk.downloader -d "$NLTK_DATA" \
    punkt stopwords wordnet omw-1.4 vader_lexicon sentiwordnet \
    averaged_perceptron_tagger maxent_ne_chunker words

COPY --chown=user:user . .

EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "2", "--timeout", "180", "src.sensecatch.app:app"]

# Tiny static server for the vLLM V1 pipeline docs (the book + interactive landings).
# Build context = this landing/ directory.
#
#   docker build -t vllm-pipeline-book docs/superpowers/landing
#   docker run -d --name vllm-book -p 8080:80 vllm-pipeline-book
#   → open http://<host>:8080/  (defaults to landing.html, the offline book)
#
# Everything is static; no network needed at runtime. See DEPLOY.md.
FROM nginx:alpine
COPY default.conf /etc/nginx/conf.d/default.conf
COPY . /usr/share/nginx/html
EXPOSE 80

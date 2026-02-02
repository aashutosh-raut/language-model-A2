from flask import Flask, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix
from model_utils import load_model, generate_text

app = Flask(__name__)

# REQUIRED for JupyterHub / Puffer reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)

# Load model once at startup
model, vocab, device = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    generated = None

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        if prompt:
            generated = generate_text(
                prompt,
                model,
                vocab,
                device,
                max_len=40,
                temperature=0.8
            )

    return render_template("index.html", generated=generated)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

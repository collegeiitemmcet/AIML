# Paste in Colab to run
streamlit run app.py &>/dev/null&
from pyngrok import ngrok
url = ngrok.connect(8501)
print("🌐 LIVE DEMO:", url)

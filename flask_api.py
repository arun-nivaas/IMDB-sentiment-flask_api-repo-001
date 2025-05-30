from flask import Flask, request, jsonify
from transformers import TFAutoModelForSequenceClassification, BertTokenizerFast
import tensorflow as tf
import constants as const

app = Flask(__name__)

tokenizer = BertTokenizerFast.from_pretrained(const.FINAL_MODEL)
model = TFAutoModelForSequenceClassification.from_pretrained(const.FINAL_MODEL)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' in request"}), 400

        text = data["text"]
        if text.strip() == "":
            return jsonify({"error": const.EMPTY}), 400
        elif len(text.strip()) < 15:
            return jsonify({"error": const.SHORT}), 400

        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        inputs.pop("token_type_ids", None)

        predictions = model(inputs)[0]
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        confidence = tf.nn.softmax(predictions, axis=1).numpy()[0][predicted_class]

        return jsonify({
            "text": text,
            "predicted_class": int(predicted_class),
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

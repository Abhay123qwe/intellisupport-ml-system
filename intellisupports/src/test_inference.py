# test_inference.py

from intellisupports.src.inference import IntelliSupportPredictor

predictor = IntelliSupportPredictor()

result = predictor.predict(
    "My payment failed but money was deducted",
    top_k=3
)

print(result)

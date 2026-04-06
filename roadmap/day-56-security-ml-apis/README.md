# Day 56: Security for ML APIs

## Learning Objectives

- Implement authentication and authorization for ML APIs
- Add rate limiting to prevent abuse and control costs
- Validate and sanitize inputs to prevent adversarial attacks
- Understand common security threats to ML systems
- Apply OWASP security principles to ML deployments

## Key Concepts

ML APIs face unique security challenges beyond traditional web applications. An exposed ML endpoint can be abused for model extraction (an attacker queries your model thousands of times to train a clone), adversarial inputs (crafted inputs that cause misclassification), and resource exhaustion (large inputs that consume GPU/memory).

**Authentication** ensures only authorized users can access your API. JWT (JSON Web Tokens) are the standard approach: the client authenticates once, receives a signed token, and includes it in subsequent requests. FastAPI's `Depends` system makes this clean. **Rate limiting** controls how many requests a client can make per time window, protecting against both abuse and accidental overuse. Libraries like `slowapi` integrate with FastAPI. **Input validation** goes beyond type checking — for ML, you need to validate input dimensions, text lengths, image sizes, and reject inputs that fall far outside the training distribution.

For production ML systems, also consider: logging all predictions for audit trails, encrypting model artifacts at rest, using HTTPS, and implementing model versioning so you can roll back if a model is compromised.

## Hands-On Exercise

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import jwt
import time

app = FastAPI()
security = HTTPBearer()
SECRET_KEY = "your-secret-key-change-in-production"

# JWT Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        if payload["exp"] < time.time():
            raise HTTPException(status_code=401, detail="Token expired")
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Input validation for ML
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)
    
    @validator("text")
    def validate_text(cls, v):
        if len(v.split()) < 3:
            raise ValueError("Text must contain at least 3 words")
        # Check for suspicious patterns
        if v.count("\x00") > 0:
            raise ValueError("Null bytes not allowed")
        return v.strip()

@app.post("/predict")
async def predict(request: PredictionRequest, user=Depends(verify_token)):
    # Authenticated and validated prediction
    return {"prediction": "positive", "user": user["sub"]}
```

## Resources

- [OWASP ML Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [FastAPI Security Docs](https://fastapi.tiangolo.com/tutorial/security/)
- [Adversarial Machine Learning Reading List](https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html)

## Next Day Preview

Tomorrow is **Portfolio Preparation** — polishing your 10 projects with documentation, demos, and presentation-ready materials.

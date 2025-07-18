from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import status
from pydantic import BaseModel, Field
import os

# PUBLIC_INTERFACE
from dotenv import load_dotenv
import openai

app = FastAPI(
    title="Tic Tac Toe Backend API",
    description="Backend API for Tic Tac Toe game, now with OpenAI-powered assistant endpoint.",
    version="0.1.0",
    openapi_tags=[
        {
            "name": "Assistant",
            "description": "Endpoints for accessing the OpenAI-powered game assistant."
        },
        {
            "name": "Health",
            "description": "Health and status endpoints."
        }
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env (including OPENAI_API_KEY)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment. Please add it to your .env file.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# PUBLIC_INTERFACE
class AssistantRequest(BaseModel):
    """Request body for the assistant endpoint."""
    message: str = Field(..., description="User's prompt or question for the assistant.")

class AssistantResponse(BaseModel):
    """Response from the assistant endpoint."""
    answer: str = Field(..., description="Assistant's reply to the user's prompt.")

@app.get("/", tags=["Health"])
def health_check():
    """Basic health check route."""
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.post("/assistant", response_model=AssistantResponse, tags=["Assistant"], summary="Ask the OpenAI-powered assistant", description="""
Submit a user question or game-related help request to the assistant. The assistant will use OpenAI's API to provide a helpful response.

- This is intended for Tic Tac Toe strategic help, general questions, or instructions.
- Your message is sent securely; the API key is never shared with clients.
""")
async def ask_assistant(request: AssistantRequest):
    """
    Process a user prompt using the OpenAI GPT-3.5 Turbo model and return the response.

    - **request**: JSON with a message string.
    - **returns**: JSON with the assistant's response.

    Raises 502 if the OpenAI API is not available.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are the helpful AI assistant for an online Tic Tac Toe game. Answer rules, strategies, and give helpful, friendly advice for the game."},
                {"role": "user", "content": request.message},
            ],
            max_tokens=256,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return AssistantResponse(answer=answer)
    except Exception as e:
        # Never leak API keys or internals to clients
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to get a response from the assistant. Please try again later."
        ) from e

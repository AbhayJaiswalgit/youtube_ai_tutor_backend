from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from bson import ObjectId

from app.core.config import settings
from app.core.database import get_database

# This tells FastAPI where the frontend should send login requests to get a token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme), db = Depends(get_database)):
    """
    Validates the JWT token in the request header and returns the database User object.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # 1. Decode the token using our secret key
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        
        # 2. Extract the user_id (we stored it as 'sub' in the auth route)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
    except jwt.PyJWTError:
        # Catches expired or tampered tokens
        raise credentials_exception

    # 3. Verify the user actually still exists in the database
    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise credentials_exception
        
    # Convert ObjectId to string for easier use in our routes
    user["_id"] = str(user["_id"])
    return user
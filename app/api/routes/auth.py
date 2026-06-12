from fastapi import APIRouter, HTTPException, Depends, status
from datetime import datetime, timezone
from app.schemas.user_schema import UserCreate, UserLogin, TokenResponse
from app.core.database import get_database
from app.core.security import get_password_hash, verify_password, create_access_token

router = APIRouter()

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, db=Depends(get_database)):
    users_collection = db["users"]
    
    # 1. Check if user already exists
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    # 2. Hash password and prepare database record
    user_data = {
        "name": user.name,
        "email": user.email,
        "hashed_password": get_password_hash(user.password),
        "created_at": datetime.now(timezone.utc)
    }
    
    # 3. Save to MongoDB
    result = await users_collection.insert_one(user_data)
    user_id = str(result.inserted_id)
    
    # 4. Generate JWT Token
    access_token = create_access_token(data={"sub": user_id})
    
    return TokenResponse(
        access_token=access_token,
        user_id=user_id,
        name=user.name
    )

@router.post("/login", response_model=TokenResponse)
async def login_user(user: UserLogin, db=Depends(get_database)):
    users_collection = db["users"]
    
    # 1. Find user by email
    db_user = await users_collection.find_one({"email": user.email})
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    # 2. Verify password
    if not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    # 3. Generate JWT Token
    user_id = str(db_user["_id"])
    access_token = create_access_token(data={"sub": user_id})
    
    return TokenResponse(
        access_token=access_token,
        user_id=user_id,
        name=db_user["name"]
    )
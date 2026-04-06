from fastapi import APIRouter, HTTPException, status
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.auth import AuthResponse, LoginRequest, SignupRequest, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse)
async def signup(payload: SignupRequest):
    existing = await User.find_one(User.email == payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password)
    )
    await user.insert()

    return AuthResponse(
        access_token=create_access_token(str(user.id)),
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at
        )
    )


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest):
    user = await User.find_one(User.email == payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    return AuthResponse(
        access_token=create_access_token(str(user.id)),
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at
        )
    )

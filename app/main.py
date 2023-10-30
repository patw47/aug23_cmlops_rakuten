from datetime import timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from authentification import get_user, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES, fake_users_db, Token, oauth2_scheme, OAuth2PasswordRequestForm, verify_password

app = FastAPI(title="API RAKUTEN")

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(fake_users_db, form_data.username)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"username": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/model/fusion/predict")
async def predict_endpoint(text: str = Form(...), image: UploadFile = UploadFile(...), current_user: dict = Depends(get_current_user)):
    image_contents = await image.read()
    with open("temp_image.jpg", "wb") as f:
        f.write(image_contents)

    # Votre code de prédiction ici...
    # prdtypecode, thematique = predict(text, "temp_image.jpg")
    # return JSONResponse(content={"prdtypecode": prdtypecode, "thematique": thematique})

    return {"message": "Prediction placeholder"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

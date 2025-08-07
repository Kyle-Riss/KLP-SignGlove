# FastAPI DataCollectionServer 스텁
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return {'message': 'Data Collection Server'}

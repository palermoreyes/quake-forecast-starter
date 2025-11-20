# src/api/database.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# URL de conexión. Usa el nombre del servicio 'db' definido en docker-compose
# El driver postgresql+asyncpg es CRÍTICO para el rendimiento asíncrono

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://quake:changeme@db:5432/quake"
)


engine = create_async_engine(DATABASE_URL, echo=False)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Dependencia para inyectar la sesión en cada request
async def get_db():
    async with async_session() as session:
        yield session
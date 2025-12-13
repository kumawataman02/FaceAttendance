import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class Database:
    def __init__(self):
        self.engine = None
        self.async_session = None
        self.is_connected = False

    async def connect(self):
        """Connect to MySQL database"""
        try:
            # Get connection parameters from environment variables
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "3306")
            db_user = os.getenv("DB_USER", "root")
            db_password = os.getenv("DB_PASSWORD", "")
            db_name = os.getenv("DB_NAME", "class24")

            # Create async engine
            connection_string = f"mysql+aiomysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

            self.engine = create_async_engine(
                connection_string,
                echo=True,  # Set to True for SQL logging during development
                pool_pre_ping=True,
                pool_recycle=3600,
                poolclass=NullPool  # Use NullPool for better async performance
            )

            # Create async session factory
            self.async_session = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            self.is_connected = True
            print(f"✅ Connected to MySQL database: {db_name}")
            return True

        except Exception as e:
            print(f"❌ Error connecting to MySQL: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from database"""
        if self.engine:
            await self.engine.dispose()
            self.is_connected = False
            print("✅ Database connection closed")

    def get_session(self) -> AsyncSession:
        """Get async database session"""
        if not self.is_connected:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self.async_session()

    async def create_tables(self):
        """Create all tables defined in Base metadata"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("✅ Tables created successfully")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            raise

    async def execute_query(self, query, params=None):
        """Execute raw query"""
        async with self.async_session() as session:
            result = await session.execute(query, params or {})
            await session.commit()
            return result

    async def fetch_one(self, query, params=None):
        """Fetch one row"""
        result = await self.execute_query(query, params)
        return result.fetchone()

    async def fetch_all(self, query, params=None):
        """Fetch all rows"""
        result = await self.execute_query(query, params)
        return result.fetchall()


# Create global database instance
database = Database()
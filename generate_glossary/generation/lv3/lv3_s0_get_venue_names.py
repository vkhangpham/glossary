import os
import sys
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from dotenv import load_dotenv
import asyncpg
from functools import lru_cache

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from aol.src.aol.db.postgres import PostgresClient

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv3.s0")

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    """Configuration for venue names extraction"""
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv3/lv3_s0_venue_names.txt")
    POSTGRESQL_URL = os.environ.get("POSTGRESQL_URL")
    BATCH_SIZE = 1000  # Number of venues to process at once
    MAX_RETRIES = 3    # Maximum number of retries for database operations
    RETRY_DELAY = 1    # Delay between retries in seconds
    POOL_MIN_SIZE = 2  # Minimum number of connections in the pool
    POOL_MAX_SIZE = 10 # Maximum number of connections in the pool

class DatabaseConnectionError(Exception):
    """Exception raised for database connection errors."""
    pass

class VenueNameError(Exception):
    """Exception raised for venue name processing errors."""
    pass

async def create_db_client() -> PostgresClient:
    """Create and initialize a PostgreSQL client with connection pooling.
    
    Returns:
        An initialized PostgreSQL client
        
    Raises:
        DatabaseConnectionError: If connection to database fails
    """
    if not Config.POSTGRESQL_URL:
        raise DatabaseConnectionError("PostgreSQL connection URL not found in environment variables")
    
    try:
        db_client = PostgresClient(
            dsn=Config.POSTGRESQL_URL,
            min_size=Config.POOL_MIN_SIZE,
            max_size=Config.POOL_MAX_SIZE
        )
        await db_client.init_pool()
        logger.info(f"Database connection pool initialized with {Config.POOL_MIN_SIZE}-{Config.POOL_MAX_SIZE} connections")
        return db_client
    except Exception as e:
        raise DatabaseConnectionError(f"Failed to create database connection pool: {str(e)}")

async def fetch_venue_names_batch(db_client: PostgresClient, offset: int, limit: int) -> List[str]:
    """Fetch a batch of venue names from the database.
    
    Args:
        db_client: PostgreSQL client
        offset: Starting position for batch
        limit: Maximum number of venues to fetch
        
    Returns:
        List of venue names in the batch
    """
    query = """
    SELECT name 
    FROM venues 
    ORDER BY name 
    LIMIT $1 OFFSET $2
    """
    
    retries = 0
    while retries < Config.MAX_RETRIES:
        try:
            data = await db_client.fetch_many(query, (limit, offset))
            
            # Extract and clean venue names
            venues = [
                venue.get("name") 
                for venue in data 
                if venue.get("name") and venue.get("name").strip()
            ]
            
            return venues
            
        except Exception as e:
            retries += 1
            if retries >= Config.MAX_RETRIES:
                logger.error(f"Failed to fetch venues after {Config.MAX_RETRIES} attempts. Last error: {str(e)}")
                raise
            
            logger.warning(f"Database query failed (attempt {retries}/{Config.MAX_RETRIES}): {str(e)}")
            await asyncio.sleep(Config.RETRY_DELAY)

async def count_total_venues(db_client: PostgresClient) -> int:
    """Count the total number of venues in the database.
    
    Args:
        db_client: PostgreSQL client
        
    Returns:
        Total venue count
    """
    query = "SELECT COUNT(*) as total FROM venues"
    
    retries = 0
    while retries < Config.MAX_RETRIES:
        try:
            result = await db_client.fetch_one(query, ())
            return result.get("total", 0)
            
        except Exception as e:
            retries += 1
            if retries >= Config.MAX_RETRIES:
                logger.error(f"Failed to count venues after {Config.MAX_RETRIES} attempts. Last error: {str(e)}")
                raise
            
            logger.warning(f"Count query failed (attempt {retries}/{Config.MAX_RETRIES}): {str(e)}")
            await asyncio.sleep(Config.RETRY_DELAY)

async def fetch_all_venue_names() -> List[str]:
    """Fetch all venue names from PostgreSQL database in batches.
    
    Returns:
        List of all venue names
        
    Raises:
        DatabaseConnectionError: If connection to database fails
        VenueNameError: If venue names cannot be processed
    """
    db_client = None
    all_venues = set()
    
    try:
        # Create database client with connection pool
        db_client = await create_db_client()
        
        # Get total count for progress tracking
        total_count = await count_total_venues(db_client)
        logger.info(f"Found {total_count} venues in database")
        
        # Process in batches
        offset = 0
        while True:
            venues = await fetch_venue_names_batch(db_client, offset, Config.BATCH_SIZE)
            
            if not venues:
                break
                
            # Add venues to result set
            all_venues.update(venues)
            logger.debug(f"Fetched batch of {len(venues)} venues (offset {offset})")
            
            # Update offset for next batch
            offset += len(venues)
            
            # Show progress
            if total_count > 0:
                progress = min(100, round(offset / total_count * 100, 1))
                logger.info(f"Progress: {progress}% ({offset}/{total_count})")
                
        logger.info(f"Completed fetching {len(all_venues)} unique venue names")
        return sorted(list(all_venues))
        
    except DatabaseConnectionError as e:
        logger.error(f"Database connection error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching venue names: {str(e)}")
        raise VenueNameError(f"Failed to fetch venue names: {str(e)}")
    finally:
        # Ensure connection pool is closed
        if db_client:
            await db_client.close()
            logger.debug("Database connection pool closed")

async def main():
    """Main execution function"""
    start_time = time.time()
    
    try:
        logger.info("Starting venue names extraction")
        
        # Fetch venue names in batches
        venues = await fetch_all_venue_names()
        logger.info(f"Successfully processed {len(venues)} unique venue names")
        
        # Create output directory if needed
        output_path = Path(Config.OUTPUT_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(Config.OUTPUT_FILE, "w", encoding='utf-8') as f:
            for venue in venues:
                f.write(f"{venue}\n")
                
        elapsed_time = time.time() - start_time
        logger.info(f"Venue names extraction completed in {elapsed_time:.2f} seconds")
        
    except VenueNameError as e:
        logger.error(f"Venue name processing error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

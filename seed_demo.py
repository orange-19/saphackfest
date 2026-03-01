import asyncio
import uuid
import sys
import os
from datetime import datetime
from sqlalchemy import text

# Add parent directory to path so we can import from db
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.database import setup_engine, get_db_session, DB_FILE

async def seed_hallucination_demo():
    print(f"Seeding demo data to {DB_FILE}...")
    setup_engine("") # Uses default SQLite path in db/database.py
    
    async with get_db_session() as session:
        # 1. Clear existing LEHENGA-001 logs to avoid clutter
        await session.execute(text("DELETE FROM governance_logs WHERE product_id = :pid"), {"pid": "LEHENGA-001"})
        
        # 2. Insert the specific hallucination record
        # Time set to 2:00 AM today
        demo_time = datetime.now().replace(hour=2, minute=0, second=0, microsecond=0).isoformat()
        
        sql = """
        INSERT INTO governance_logs (
            request_id, product_id, bot_id, bot_reasoning, 
            original_price, proposed_price, price_delta_pct, 
            status, layer1_result, ai_reasoning, ai_confidence_score,
            created_at, updated_at
        ) VALUES (
            :rid, :pid, :bid, :breason, 
            :orig, :prop, :delta, 
            :status, :l1, :aireason, :score,
            :cat, :uat
        )
        """
        
        params = {
            "rid": str(uuid.uuid4()),
            "pid": "LEHENGA-001",
            "bid": "Worker_AI_04",
            "breason": "Aggressive clearance pricing for slow-moving inventory.",
            "orig": 15000.0,
            "prop": 1.0,
            "delta": -99.99,
            "status": "rejected",
            "l1": "Policy 1 Violation: Human session logged out. AI Hallucination detected.",
            "aireason": "Critical anomaly: AI attempted 99.99% discount while no authorized human supervisor was active.",
            "score": 0.0,
            "cat": demo_time,
            "uat": demo_time
        }
        
        await session.execute(text(sql), params)
        print("✅ Demo record for LEHENGA-001 (Hallucination) inserted successfully.")

if __name__ == "__main__":
    asyncio.run(seed_hallucination_demo())

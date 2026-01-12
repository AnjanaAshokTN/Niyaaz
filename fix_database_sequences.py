"""
Fix PostgreSQL database sequences that are out of sync
This script resets sequences to be higher than the maximum existing ID
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME', 'sakshiai')
db_user = os.getenv('DB_USER', 'postgres')
db_password = os.getenv('DB_PASSWORD', '')

if not db_password:
    print("ERROR: DB_PASSWORD not set in environment variables")
    print("Please set DB_PASSWORD in your .env file or environment")
    sys.exit(1)

try:
    import psycopg2
    from urllib.parse import quote_plus
    
    # URL-encode password
    encoded_password = quote_plus(db_password)
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    cur = conn.cursor()
    
    print("Connected to PostgreSQL database")
    print(f"Database: {db_name}@{db_host}:{db_port}")
    print("\nFixing database sequences...\n")
    
    # List of tables and their sequence names
    tables_to_fix = [
        ('cash_snapshots', 'cash_snapshots_id_seq'),
        ('alert_gifs', 'alert_gifs_id_seq'),
        ('queue_violations', 'queue_violations_id_seq'),
        ('ppe_alerts', 'ppe_alerts_id_seq'),
        ('dresscode_violations', 'dresscode_violations_id_seq'),
        ('table_service_violations', 'table_service_violations_id_seq'),
        ('fall_snapshots', 'fall_snapshots_id_seq'),
        ('smoking_snapshots', 'smoking_snapshots_id_seq'),
        ('material_theft_snapshots', 'material_theft_snapshots_id_seq'),
    ]
    
    fixed_count = 0
    
    for table_name, sequence_name in tables_to_fix:
        try:
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table_name,))
            
            if not cur.fetchone()[0]:
                print(f"  ⏭️  Table '{table_name}' does not exist, skipping...")
                continue
            
            # Get maximum ID from table
            cur.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table_name};")
            max_id = cur.fetchone()[0]
            
            # Get current sequence value
            cur.execute(f"SELECT last_value FROM {sequence_name};")
            current_seq = cur.fetchone()[0]
            
            if max_id >= current_seq:
                # Reset sequence to be higher than max_id
                new_seq_value = max_id + 1
                cur.execute(f"SELECT setval('{sequence_name}', {new_seq_value}, false);")
                conn.commit()
                print(f"  ✅ Fixed '{table_name}': sequence reset from {current_seq} to {new_seq_value} (max_id={max_id})")
                fixed_count += 1
            else:
                print(f"  ✓ '{table_name}': sequence OK (current={current_seq}, max_id={max_id})")
                
        except Exception as e:
            print(f"  ❌ Error fixing '{table_name}': {e}")
            conn.rollback()
    
    print(f"\n✅ Fixed {fixed_count} sequence(s)")
    print("\nDatabase sequences are now synchronized!")
    
    cur.close()
    conn.close()
    
except ImportError:
    print("ERROR: psycopg2 not installed")
    print("Please install it: pip install psycopg2-binary")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


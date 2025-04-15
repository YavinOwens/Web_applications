import os
import sys
import unittest
import sqlite3

class DatabaseConnectionTests(unittest.TestCase):
    def setUp(self):
        # Paths to test databases
        self.sqlite_path = os.path.join(os.path.dirname(__file__), '../../../crud_project/db.sqlite3')

    def test_sqlite_connection(self):
        """Test connection to SQLite database."""
        try:
            # Attempt to connect directly using sqlite3
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Try a simple query
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5;")
            tables = cursor.fetchall()
            
            # Basic assertions
            self.assertIsNotNone(tables)
            self.assertTrue(isinstance(tables, list))
            
            # Close the connection
            cursor.close()
            conn.close()
        except Exception as e:
            self.fail(f"SQLite connection failed: {str(e)}")

    def test_database_file_exists(self):
        """Verify the database file exists."""
        self.assertTrue(
            os.path.exists(self.sqlite_path), 
            f"Database file not found at {self.sqlite_path}"
        )
        
        # Check file is readable
        try:
            with open(self.sqlite_path, 'rb') as f:
                f.read(1)  # Try to read a single byte
        except Exception as e:
            self.fail(f"Cannot read database file: {str(e)}")

    def test_database_file_size(self):
        """Check database file size."""
        file_size = os.path.getsize(self.sqlite_path)
        
        # Ensure file is not empty
        self.assertGreater(file_size, 0, "Database file is empty")
        
        # Optional: Set a reasonable max size (adjust as needed)
        self.assertLess(file_size, 100 * 1024 * 1024, "Database file is unexpectedly large")

    def test_database_connection_performance(self):
        """Basic performance test for database connections."""
        import time
        
        # SQLite3 performance
        start_time = time.time()
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Perform multiple quick queries
            for _ in range(10):
                cursor.execute("SELECT 1;")
            
            cursor.close()
            conn.close()
        except Exception as e:
            self.fail(f"SQLite3 performance test failed: {str(e)}")
        
        sqlite_time = time.time() - start_time
        
        # Print performance for debugging
        print(f"SQLite3 Connection Performance: {sqlite_time:.4f} seconds")
        
        # Assert reasonable connection time (adjust threshold as needed)
        self.assertLess(sqlite_time, 5.0)

    def test_optional_database_libraries(self):
        """Check availability of optional database libraries."""
        optional_libraries = [
            'psycopg2',
            'mysql.connector'
        ]
        
        for library in optional_libraries:
            try:
                __import__(library)
                print(f"{library} is available")
            except ImportError:
                print(f"{library} is not installed")

    def test_sqlite_query_capabilities(self):
        """Test basic query capabilities of SQLite."""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Try various query types
            cursor.execute("SELECT COUNT(*) FROM sqlite_master;")
            table_count = cursor.fetchone()[0]
            
            # Verify we can get table count
            self.assertIsNotNone(table_count)
            self.assertGreaterEqual(table_count, 0)
            
            # Try a more complex query if possible
            try:
                cursor.execute("PRAGMA table_info(sqlite_master);")
                table_info = cursor.fetchall()
                self.assertTrue(len(table_info) > 0)
            except sqlite3.OperationalError:
                # Some SQLite configurations might restrict this
                pass
            
            cursor.close()
            conn.close()
        except Exception as e:
            self.fail(f"SQLite query capabilities test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 
"""
Database manager for MySQL operations.
"""

import os
import json
import mysql.connector
from typing import Dict, Any, List, Optional, Tuple, Union
from dotenv import load_dotenv
import sqlite3  # For in-memory database fallback

# Load environment variables
load_dotenv()

class DatabaseManager:
    """Class for handling database operations."""
    
    def __init__(self, use_in_memory: bool = True):
        """
        Initialize the database connection.
        
        Args:
            use_in_memory: Whether to use an in-memory SQLite database instead of MySQL
        """
        self.connection = None
        self.connected = False
        self.use_in_memory = use_in_memory
        
        if use_in_memory:
            self._connect_sqlite()
        else:
            self.connect()
    
    def _connect_sqlite(self) -> bool:
        """
        Connect to an in-memory SQLite database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = sqlite3.connect(':memory:')
            self.cursor = self.connection.cursor()
            self.connected = True
            self._create_sqlite_schema()
            return True
        except sqlite3.Error as err:
            print(f"SQLite connection error: {err}")
            self.connected = False
            return False
    
    def _create_sqlite_schema(self) -> None:
        """Create the SQLite database schema."""
        # Images table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            width INTEGER,
            height INTEGER,
            color_space TEXT,
            file_format TEXT
        )
        ''')
        
        # Enhancements table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS enhancements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            enhancement_type TEXT NOT NULL,
            parameters TEXT,  # JSON string
            creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            output_filename TEXT,
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
        ''')
        
        # Metrics table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            enhancement_id INTEGER,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            FOREIGN KEY (enhancement_id) REFERENCES enhancements(id)
        )
        ''')
        
        self.connection.commit()
    
    def connect(self) -> bool:
        """
        Connect to the MySQL database.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.use_in_memory:
            return self._connect_sqlite()
            
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'image_user'),
                password=os.getenv('DB_PASSWORD', ''),
                database=os.getenv('DB_NAME', 'image_enhancement')
            )
            self.cursor = self.connection.cursor(dictionary=True)
            self.connected = True
            return True
        except mysql.connector.Error as err:
            print(f"MySQL connection error: {err}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self.connection and self.connected:
            self.connection.close()
            self.connected = False
    
    def create_schema(self) -> bool:
        """
        Create the database schema if it doesn't exist.
        
        Returns:
            True if successful, False otherwise
        """
        if self.use_in_memory:
            return True  # Schema already created in _connect_sqlite
            
        if not self.connected and not self.connect():
            return False
            
        try:
            # Images table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_size INT,
                width INT,
                height INT,
                color_space VARCHAR(20),
                file_format VARCHAR(10)
            )
            ''')
            
            # Enhancements table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhancements (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_id INT,
                enhancement_type VARCHAR(50) NOT NULL,
                parameters JSON,
                creation_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                output_filename VARCHAR(255),
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
            ''')
            
            # Metrics table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                enhancement_id INT,
                metric_name VARCHAR(50) NOT NULL,
                metric_value FLOAT,
                FOREIGN KEY (enhancement_id) REFERENCES enhancements(id)
            )
            ''')
            
            self.connection.commit()
            return True
        except mysql.connector.Error as err:
            print(f"Error creating schema: {err}")
            return False
    
    # Image operations
    def add_image(self, image_data: Dict[str, Any]) -> Optional[int]:
        """
        Add a new image to the database.
        
        Args:
            image_data: Dictionary with image metadata
                {
                    'filename': str,
                    'file_size': int,
                    'width': int,
                    'height': int,
                    'color_space': str,
                    'file_format': str
                }
                
        Returns:
            Image ID if successful, None otherwise
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            if self.use_in_memory:
                # SQLite version
                query = '''
                INSERT INTO images (filename, file_size, width, height, color_space, file_format)
                VALUES (?, ?, ?, ?, ?, ?)
                '''
                self.cursor.execute(query, (
                    image_data.get('filename', ''),
                    image_data.get('file_size', 0),
                    image_data.get('width', 0),
                    image_data.get('height', 0),
                    image_data.get('color_space', ''),
                    image_data.get('file_format', '')
                ))
                self.connection.commit()
                return self.cursor.lastrowid
            else:
                # MySQL version
                query = '''
                INSERT INTO images (filename, file_size, width, height, color_space, file_format)
                VALUES (%s, %s, %s, %s, %s, %s)
                '''
                self.cursor.execute(query, (
                    image_data.get('filename', ''),
                    image_data.get('file_size', 0),
                    image_data.get('width', 0),
                    image_data.get('height', 0),
                    image_data.get('color_space', ''),
                    image_data.get('file_format', '')
                ))
                self.connection.commit()
                return self.cursor.lastrowid
        except (mysql.connector.Error, sqlite3.Error) as err:
            print(f"Error adding image: {err}")
            return None
    
    def get_image(self, image_id: int) -> Optional[Dict[str, Any]]:
        """
        Get image metadata by ID.
        
        Args:
            image_id: Image ID
            
        Returns:
            Dictionary with image metadata if found, None otherwise
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            if self.use_in_memory:
                # SQLite version
                query = "SELECT * FROM images WHERE id = ?"
                self.cursor.execute(query, (image_id,))
                row = self.cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in self.cursor.description]
                    return dict(zip(columns, row))
                return None
            else:
                # MySQL version
                query = "SELECT * FROM images WHERE id = %s"
                self.cursor.execute(query, (image_id,))
                return self.cursor.fetchone()
        except (mysql.connector.Error, sqlite3.Error) as err:
            print(f"Error getting image: {err}")
            return None
    
    def get_all_images(self) -> List[Dict[str, Any]]:
        """
        Get all images from the database.
        
        Returns:
            List of dictionaries with image metadata
        """
        if not self.connected:
            if not self.connect():
                return []
        
        try:
            if self.use_in_memory:
                # SQLite version
                query = "SELECT * FROM images ORDER BY upload_date DESC"
                self.cursor.execute(query)
                rows = self.cursor.fetchall()
                if rows:
                    columns = [desc[0] for desc in self.cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
                return []
            else:
                # MySQL version
                query = "SELECT * FROM images ORDER BY upload_date DESC"
                self.cursor.execute(query)
                return self.cursor.fetchall()
        except (mysql.connector.Error, sqlite3.Error) as err:
            print(f"Error getting all images: {err}")
            return []
    
    # Enhancement operations
    def add_enhancement(self, enhancement_data: Dict[str, Any]) -> Optional[int]:
        """
        Add a new enhancement to the database.
        
        Args:
            enhancement_data: Dictionary with enhancement data
                {
                    'image_id': int,
                    'enhancement_type': str,
                    'parameters': dict,
                    'output_filename': str
                }
                
        Returns:
            Enhancement ID if successful, None otherwise
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            # Convert parameters dict to JSON string
            parameters = enhancement_data.get('parameters', {})
            if isinstance(parameters, dict):
                parameters_json = json.dumps(parameters)
            else:
                parameters_json = parameters
            
            if self.use_in_memory:
                # SQLite version
                query = '''
                INSERT INTO enhancements (image_id, enhancement_type, parameters, output_filename)
                VALUES (?, ?, ?, ?)
                '''
                self.cursor.execute(query, (
                    enhancement_data.get('image_id', 0),
                    enhancement_data.get('enhancement_type', ''),
                    parameters_json,
                    enhancement_data.get('output_filename', '')
                ))
                self.connection.commit()
                return self.cursor.lastrowid
            else:
                # MySQL version
                query = '''
                INSERT INTO enhancements (image_id, enhancement_type, parameters, output_filename)
                VALUES (%s, %s, %s, %s)
                '''
                self.cursor.execute(query, (
                    enhancement_data.get('image_id', 0),
                    enhancement_data.get('enhancement_type', ''),
                    parameters_json,
                    enhancement_data.get('output_filename', '')
                ))
                self.connection.commit()
                return self.cursor.lastrowid
        except (mysql.connector.Error, sqlite3.Error) as err:
            print(f"Error adding enhancement: {err}")
            return None
    
    def get_enhancements_for_image(self, image_id: int) -> List[Dict[str, Any]]:
        """
        Get all enhancements for a specific image.
        
        Args:
            image_id: Image ID
            
        Returns:
            List of dictionaries with enhancement data
        """
        if not self.connected:
            if not self.connect():
                return []
        
        try:
            if self.use_in_memory:
                # SQLite version
                query = "SELECT * FROM enhancements WHERE image_id = ? ORDER BY creation_date DESC"
                self.cursor.execute(query, (image_id,))
                rows = self.cursor.fetchall()
                if rows:
                    columns = [desc[0] for desc in self.cursor.description]
                    result = []
                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        # Parse JSON parameters
                        if 'parameters' in row_dict and row_dict['parameters']:
                            try:
                                row_dict['parameters'] = json.loads(row_dict['parameters'])
                            except json.JSONDecodeError:
                                pass
                        result.append(row_dict)
                    return result
                return []
            else:
                # MySQL version
                query = "SELECT * FROM enhancements WHERE image_id = %s ORDER BY creation_date DESC"
                self.cursor.execute(query, (image_id,))
                return self.cursor.fetchall()
        except (mysql.connector.Error, sqlite3.Error) as err:
            print(f"Error getting enhancements: {err}")
            return []
    
    # Metrics operations
    def add_metrics(self, metrics_data: List[Dict[str, Any]]) -> bool:
        """
        Add multiple metrics to the database.
        
        Args:
            metrics_data: List of dictionaries with metric data
                [
                    {
                        'enhancement_id': int,
                        'metric_name': str,
                        'metric_value': float
                    },
                    ...
                ]
                
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            for metric in metrics_data:
                if self.use_in_memory:
                    # SQLite version
                    query = '''
                    INSERT INTO metrics (enhancement_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                    '''
                    self.cursor.execute(query, (
                        metric.get('enhancement_id', 0),
                        metric.get('metric_name', ''),
                        metric.get('metric_value', 0.0)
                    ))
                else:
                    # MySQL version
                    query = '''
                    INSERT INTO metrics (enhancement_id, metric_name, metric_value)
                    VALUES (%s, %s, %s)
                    '''
                    self.cursor.execute(query, (
                        metric.get('enhancement_id', 0),
                        metric.get('metric_name', ''),
                        metric.get('metric_value', 0.0)
                    ))
            
            self.connection.commit()
            return True
        except (mysql.connector.Error, sqlite3.Error) as err:
            print(f"Error adding metrics: {err}")
            return False
    
    def get_metrics_for_enhancement(self, enhancement_id: int) -> List[Dict[str, Any]]:
        """
        Get all metrics for a specific enhancement.
        
        Args:
            enhancement_id: Enhancement ID
            
        Returns:
            List of dictionaries with metric data
        """
        if not self.connected:
            if not self.connect():
                return []
        
        try:
            if self.use_in_memory:
                # SQLite version
                query = "SELECT * FROM metrics WHERE enhancement_id = ?"
                self.cursor.execute(query, (enhancement_id,))
                rows = self.cursor.fetchall()
                if rows:
                    columns = [desc[0] for desc in self.cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
                return []
            else:
                # MySQL version
                query = "SELECT * FROM metrics WHERE enhancement_id = %s"
                self.cursor.execute(query, (enhancement_id,))
                return self.cursor.fetchall()
        except (mysql.connector.Error, sqlite3.Error) as err:
            print(f"Error getting metrics: {err}")
            return []

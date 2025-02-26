"""
MySQL connector module.

This module provides functionality for connecting to a MySQL database.
"""

import mysql.connector
from mysql.connector import Error
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import json


class MySQLConnector:
    """Class for connecting to a MySQL database."""

    def __init__(self, host: str = "localhost", user: str = "root", password: str = "",
                database: str = "blood_vessel_db", port: int = 3306):
        """
        Initialize the MySQL connector.

        Args:
            host: The database host.
            user: The database user.
            password: The database password.
            database: The database name.
            port: The database port.
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        self.cursor = None

    def connect(self) -> bool:
        """
        Connect to the MySQL database.

        Returns:
            bool: True if the connection was successful, False otherwise.
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port
            )
            
            if self.connection.is_connected():
                # Create the database if it doesn't exist
                self.cursor = self.connection.cursor()
                self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                self.cursor.execute(f"USE {self.database}")
                return True
            
            return False
        
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            return False

    def disconnect(self):
        """Disconnect from the MySQL database."""
        if self.connection is not None and self.connection.is_connected():
            if self.cursor is not None:
                self.cursor.close()
            self.connection.close()
            self.connection = None
            self.cursor = None

    def is_connected(self) -> bool:
        """
        Check if the connector is connected to the database.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.connection is not None and self.connection.is_connected()

    def execute_query(self, query: str, params: Tuple = None) -> bool:
        """
        Execute a query.

        Args:
            query: The query to execute.
            params: The parameters for the query.

        Returns:
            bool: True if the query was executed successfully, False otherwise.
        """
        if not self.is_connected():
            print("Error: Not connected to the database.")
            return False

        try:
            self.cursor.execute(query, params)
            self.connection.commit()
            return True
        
        except Error as e:
            print(f"Error executing query: {e}")
            return False

    def fetch_all(self, query: str, params: Tuple = None) -> List[Tuple]:
        """
        Fetch all rows from a query.

        Args:
            query: The query to execute.
            params: The parameters for the query.

        Returns:
            List[Tuple]: The fetched rows.
        """
        if not self.is_connected():
            print("Error: Not connected to the database.")
            return []

        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        
        except Error as e:
            print(f"Error fetching data: {e}")
            return []

    def fetch_one(self, query: str, params: Tuple = None) -> Optional[Tuple]:
        """
        Fetch one row from a query.

        Args:
            query: The query to execute.
            params: The parameters for the query.

        Returns:
            Optional[Tuple]: The fetched row, or None if no row was fetched.
        """
        if not self.is_connected():
            print("Error: Not connected to the database.")
            return None

        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchone()
        
        except Error as e:
            print(f"Error fetching data: {e}")
            return None

    def create_tables(self) -> bool:
        """
        Create the database tables.

        Returns:
            bool: True if the tables were created successfully, False otherwise.
        """
        if not self.is_connected():
            print("Error: Not connected to the database.")
            return False

        try:
            # Create the models table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255),
                    filename VARCHAR(255),
                    surface_area FLOAT,
                    volume FLOAT,
                    centerline_length FLOAT,
                    num_branch_points INT,
                    num_endpoints INT,
                    num_segments INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    properties JSON
                )
            """)

            # Create the nodes table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id VARCHAR(255),
                    model_id VARCHAR(255),
                    node_type VARCHAR(50),
                    position_x FLOAT,
                    position_y FLOAT,
                    position_z FLOAT,
                    properties JSON,
                    PRIMARY KEY (node_id, model_id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
                )
            """)

            # Create the segments table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id INT,
                    model_id VARCHAR(255),
                    start_node_id VARCHAR(255),
                    end_node_id VARCHAR(255),
                    length FLOAT,
                    diameter FLOAT,
                    tortuosity FLOAT,
                    properties JSON,
                    PRIMARY KEY (segment_id, model_id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE,
                    FOREIGN KEY (start_node_id, model_id) REFERENCES nodes(node_id, model_id) ON DELETE CASCADE,
                    FOREIGN KEY (end_node_id, model_id) REFERENCES nodes(node_id, model_id) ON DELETE CASCADE
                )
            """)

            # Create the segment_points table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS segment_points (
                    segment_id INT,
                    model_id VARCHAR(255),
                    point_index INT,
                    position_x FLOAT,
                    position_y FLOAT,
                    position_z FLOAT,
                    PRIMARY KEY (segment_id, model_id, point_index),
                    FOREIGN KEY (segment_id, model_id) REFERENCES segments(segment_id, model_id) ON DELETE CASCADE
                )
            """)

            # Create the cross_sections table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS cross_sections (
                    cross_section_id INT AUTO_INCREMENT PRIMARY KEY,
                    model_id VARCHAR(255),
                    segment_id INT,
                    position_x FLOAT,
                    position_y FLOAT,
                    position_z FLOAT,
                    area FLOAT,
                    perimeter FLOAT,
                    equivalent_diameter FLOAT,
                    circularity FLOAT,
                    properties JSON,
                    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE,
                    FOREIGN KEY (segment_id, model_id) REFERENCES segments(segment_id, model_id) ON DELETE CASCADE
                )
            """)

            # Create the bifurcation_angles table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS bifurcation_angles (
                    bifurcation_id INT AUTO_INCREMENT PRIMARY KEY,
                    model_id VARCHAR(255),
                    node_id VARCHAR(255),
                    segment1_id INT,
                    segment2_id INT,
                    angle FLOAT,
                    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE,
                    FOREIGN KEY (node_id, model_id) REFERENCES nodes(node_id, model_id) ON DELETE CASCADE,
                    FOREIGN KEY (segment1_id, model_id) REFERENCES segments(segment_id, model_id) ON DELETE CASCADE,
                    FOREIGN KEY (segment2_id, model_id) REFERENCES segments(segment_id, model_id) ON DELETE CASCADE
                )
            """)

            self.connection.commit()
            return True
        
        except Error as e:
            print(f"Error creating tables: {e}")
            return False

    def drop_tables(self) -> bool:
        """
        Drop the database tables.

        Returns:
            bool: True if the tables were dropped successfully, False otherwise.
        """
        if not self.is_connected():
            print("Error: Not connected to the database.")
            return False

        try:
            # Drop the tables in reverse order of creation to avoid foreign key constraints
            self.cursor.execute("DROP TABLE IF EXISTS bifurcation_angles")
            self.cursor.execute("DROP TABLE IF EXISTS cross_sections")
            self.cursor.execute("DROP TABLE IF EXISTS segment_points")
            self.cursor.execute("DROP TABLE IF EXISTS segments")
            self.cursor.execute("DROP TABLE IF EXISTS nodes")
            self.cursor.execute("DROP TABLE IF EXISTS models")
            
            self.connection.commit()
            return True
        
        except Error as e:
            print(f"Error dropping tables: {e}")
            return False

    @staticmethod
    def from_env_file(env_file: str = ".env") -> 'MySQLConnector':
        """
        Create a MySQL connector from an environment file.

        Args:
            env_file: The path to the environment file.

        Returns:
            MySQLConnector: The created MySQL connector.
        """
        # Default values
        host = "localhost"
        user = "root"
        password = ""
        database = "blood_vessel_db"
        port = 3306

        # Read the environment file
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        if key == "MYSQL_HOST":
                            host = value
                        elif key == "MYSQL_USER":
                            user = value
                        elif key == "MYSQL_PASSWORD":
                            password = value
                        elif key == "MYSQL_DATABASE":
                            database = value
                        elif key == "MYSQL_PORT":
                            port = int(value)

        return MySQLConnector(host=host, user=user, password=password, database=database, port=port)

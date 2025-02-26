# Database Setup

This document describes how to set up the MySQL database for the Image Enhancement System.

## Requirements

- MySQL Server 5.7 or higher
- MySQL Connector for Python

## Setup Steps

1. Install MySQL Server if not already installed
2. Create a new database:
   ```sql
   CREATE DATABASE image_enhancement;
   ```
3. Create a user with appropriate permissions:
   ```sql
   CREATE USER 'image_user'@'localhost' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON image_enhancement.* TO 'image_user'@'localhost';
   FLUSH PRIVILEGES;
   ```
4. Create the necessary tables (see schema below)
5. Update the `.env` file with your database credentials

## Database Schema

```sql
-- Images table to store metadata about original images
CREATE TABLE images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    file_size INT,
    width INT,
    height INT,
    color_space VARCHAR(20),
    file_format VARCHAR(10)
);

-- Enhancements table to store information about applied enhancements
CREATE TABLE enhancements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_id INT,
    enhancement_type VARCHAR(50) NOT NULL,
    parameters JSON,
    creation_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    output_filename VARCHAR(255),
    FOREIGN KEY (image_id) REFERENCES images(id)
);

-- Metrics table to store image quality metrics
CREATE TABLE metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    enhancement_id INT,
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT,
    FOREIGN KEY (enhancement_id) REFERENCES enhancements(id)
);
```

-- MySQL database schema for Image Enhancement System

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

-- Sample queries

-- Get all images with their metadata
SELECT * FROM images ORDER BY upload_date DESC;

-- Get all enhancements for a specific image
SELECT e.* FROM enhancements e
JOIN images i ON e.image_id = i.id
WHERE i.id = [image_id]
ORDER BY e.creation_date DESC;

-- Get metrics for a specific enhancement
SELECT m.* FROM metrics m
JOIN enhancements e ON m.enhancement_id = e.id
WHERE e.id = [enhancement_id];

-- Get enhancement statistics
SELECT 
    enhancement_type,
    COUNT(*) as count,
    AVG(
        (SELECT AVG(metric_value) FROM metrics 
         WHERE enhancement_id = enhancements.id AND metric_name = 'psnr')
    ) as avg_psnr
FROM enhancements
GROUP BY enhancement_type;
